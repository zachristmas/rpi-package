//! Z.ai GLM Coding Plan policy belongs to this optional package. Wire parsing,
//! retries and the native streaming ABI use the maintained provider protocol.
//!
//! Two providers share the request logic, as in Pi (badlogic/pi-mono): `zai`
//! for the global plan on api.z.ai and `zai-coding-cn` for the China plan on
//! open.bigmodel.cn. The catalogs are every tool-capable model on models.dev
//! (`zai-coding-plan` and `zhipuai-coding-plan`), with `thinking`,
//! `reasoning_effort` and `tool_stream` set per model. Plan keys are bound to
//! one platform, so a key from one endpoint fails on the other with code 1113.
use rpi_ai::{
    model::{MaxTokensField, OpenaiCompletionsCompat, StreamingProtocolCompat},
    types::InputModality,
    Api, AssistantMessageEventStream, Context, Model, Provider, SimpleStreamOptions, ThinkingLevel,
};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

pub const PROVIDER_ID: &str = "zai";
pub const CN_PROVIDER_ID: &str = "zai-coding-cn";
pub const DEFAULT_MODEL_ID: &str = "glm-5.3";
pub const FLASH_MODEL_ID: &str = "glm-5.3-flash";

/// One Coding Plan platform. The general (pay-per-token) APIs live under
/// `/api/paas/v4` and do not consume plan quota.
pub struct Variant {
    id: &'static str,
    base_url: &'static str,
    env: &'static str,
    specs: &'static [Spec],
}

pub const GLOBAL: Variant = Variant {
    id: PROVIDER_ID,
    base_url: "https://api.z.ai/api/coding/paas/v4",
    env: "ZAI_API_KEY",
    specs: GLOBAL_SPECS,
};

pub const CHINA: Variant = Variant {
    id: CN_PROVIDER_ID,
    base_url: "https://open.bigmodel.cn/api/coding/paas/v4",
    env: "ZAI_CODING_CN_API_KEY",
    specs: CHINA_SPECS,
};

/// Thinking behaviour differs by generation. GLM-5.3 always thinks and accepts
/// only `reasoning_effort`; GLM-5.2 and earlier can switch thinking off.
struct Spec {
    id: &'static str,
    name: &'static str,
    image: bool,
    context: u64,
    /// Highest answer allowance the server accepts.
    max_output: u64,
    /// Thinking can be turned off with `thinking.type = "disabled"`.
    toggle: bool,
    /// Accepted `reasoning_effort` values, lowest first. Empty means the model
    /// takes no effort parameter.
    efforts: &'static [&'static str],
}

const LOW_HIGH_MAX: &[&str] = &["low", "high", "max"];
const HIGH_MAX: &[&str] = &["high", "max"];

const fn glm53(id: &'static str, name: &'static str, image: bool) -> Spec {
    Spec {
        id,
        name,
        image,
        context: 1_000_000,
        max_output: 131_072,
        toggle: false,
        efforts: LOW_HIGH_MAX,
    }
}

/// Order matters: the first entry is the default. Models marked deprecated on
/// models.dev stay listed because the Coding Plan still routes them.
const GLOBAL_SPECS: &[Spec] = &[
    glm53(DEFAULT_MODEL_ID, "GLM-5.3", false),
    glm53("glm-5.3-highspeed", "GLM-5.3 Highspeed", false),
    glm53(FLASH_MODEL_ID, "GLM-5.3 Flash", true),
    Spec {
        id: "glm-5.2",
        name: "GLM-5.2 (deprecated)",
        image: false,
        context: 1_000_000,
        max_output: 131_072,
        toggle: true,
        efforts: HIGH_MAX,
    },
    Spec {
        id: "glm-5.2-highspeed",
        name: "GLM-5.2 Highspeed (deprecated)",
        image: false,
        context: 1_000_000,
        max_output: 131_072,
        toggle: true,
        efforts: HIGH_MAX,
    },
    Spec {
        id: "glm-5-turbo",
        name: "GLM-5 Turbo (deprecated)",
        image: false,
        context: 200_000,
        max_output: 131_072,
        toggle: true,
        efforts: &[],
    },
    Spec {
        id: "glm-4.7",
        name: "GLM-4.7 (deprecated)",
        image: false,
        context: 204_800,
        max_output: 131_072,
        toggle: true,
        efforts: &[],
    },
];

const CHINA_SPECS: &[Spec] = &[
    glm53(DEFAULT_MODEL_ID, "GLM-5.3", false),
    glm53("glm-5.3-highspeed", "GLM-5.3 Highspeed", false),
    glm53(FLASH_MODEL_ID, "GLM-5.3 Flash", true),
    Spec {
        id: "glm-4.6v",
        name: "GLM-4.6V (deprecated)",
        image: true,
        context: 128_000,
        max_output: 32_768,
        toggle: true,
        efforts: &[],
    },
];

fn spec(specs: &'static [Spec], id: &str) -> Option<&'static Spec> {
    specs.iter().find(|spec| spec.id == id)
}

fn compatibility() -> OpenaiCompletionsCompat {
    OpenaiCompletionsCompat {
        supports_store: Some(false),
        supports_developer_role: Some(false),
        supports_strict_mode: Some(false),
        supports_reasoning_effort: Some(true),
        // Tool turns must return the unmodified `reasoning_content`.
        requires_reasoning_content: Some(true),
        // Z.ai does not document whether hidden reasoning counts toward
        // `max_tokens`. Reserving the thinking budget on top of the answer
        // allowance only raises the requested ceiling, and avoids an empty
        // `finish_reason: "length"` answer if it does.
        reasoning_counts_toward_max_tokens: Some(true),
        max_tokens_field: Some(MaxTokensField::MaxTokens),
        ..Default::default()
    }
}

pub fn models_for(variant: &Variant) -> Vec<Model> {
    variant
        .specs
        .iter()
        .map(|spec| {
            let mut model = Model::new(
                spec.id,
                spec.name,
                Api::OpenaiCompletions,
                variant.id,
                variant.base_url,
            );
            model.reasoning = true;
            if spec.image {
                model.input = vec![InputModality::Text, InputModality::Image];
            }
            model.context_window = spec.context;
            // The visible answer's allowance; a per-request default, not the
            // server maximum.
            model.max_tokens = spec.max_output.min(16_384);
            model.thinking_level_map = Some(BTreeMap::from([
                (ThinkingLevel::Xhigh, Some("high".into())),
                (ThinkingLevel::Max, Some("max".into())),
            ]));
            model.compat = Some(StreamingProtocolCompat::OpenaiCompletions(compatibility()));
            // The Coding Plan bills in plan credits, not per token, so no cost
            // rates are declared. Users can add estimates in models.json.
            model
        })
        .collect()
}

/// Global catalog, kept for callers of the original single-provider API.
pub fn models() -> Vec<Model> {
    models_for(&GLOBAL)
}

/// The effort a host thinking level asks for, before clamping to the model.
fn wanted_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off | ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Max => "max",
        ThinkingLevel::Medium | ThinkingLevel::High | ThinkingLevel::Xhigh => "high",
    }
}

fn rank(effort: &str) -> usize {
    ["low", "high", "max"]
        .iter()
        .position(|candidate| *candidate == effort)
        .unwrap_or(0)
}

/// Smallest supported effort at or above the request, else the highest one.
fn clamp_effort(wanted: &str, supported: &[&'static str]) -> Option<&'static str> {
    supported
        .iter()
        .copied()
        .find(|effort| rank(effort) >= rank(wanted))
        .or_else(|| supported.last().copied())
}

/// `thinking` and `reasoning_effort` request fields for one request. Unknown
/// (user-defined) model IDs get the GLM-5.3 rules, which are the newest.
fn thinking_params(
    specs: &'static [Spec],
    model_id: &str,
    level: ThinkingLevel,
) -> (serde_json::Value, Option<&'static str>) {
    let (toggle, efforts) =
        spec(specs, model_id).map_or((false, LOW_HIGH_MAX), |s| (s.toggle, s.efforts));
    if toggle && level == ThinkingLevel::Off {
        return (json!({"type": "disabled"}), None);
    }
    // Preserved thinking, as Pi sends it: reasoning is replayed unmodified.
    let thinking = json!({"type": "enabled", "clear_thinking": false});
    (thinking, clamp_effort(wanted_effort(level), efforts))
}

/// No process environment is read here; the host supplies scoped credentials.
#[derive(Clone)]
pub struct ZaiProvider {
    variant: &'static Variant,
    api_key: Option<String>,
    models: Vec<Model>,
}

impl ZaiProvider {
    pub fn new(api_key: Option<String>) -> Self {
        Self::for_variant(&GLOBAL, api_key)
    }

    pub fn china(api_key: Option<String>) -> Self {
        Self::for_variant(&CHINA, api_key)
    }

    fn for_variant(variant: &'static Variant, api_key: Option<String>) -> Self {
        Self {
            variant,
            api_key,
            models: models_for(variant),
        }
    }
}

#[async_trait::async_trait]
impl Provider for ZaiProvider {
    fn id(&self) -> &str {
        self.variant.id
    }
    fn models(&self) -> &[Model] {
        &self.models
    }

    async fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let mut model = model.clone();
        model.provider = self.variant.id.into();
        if model.base_url.is_empty() {
            model.base_url = self.variant.base_url.into();
        }
        let compat = model
            .compat
            .get_or_insert_with(|| StreamingProtocolCompat::OpenaiCompletions(compatibility()));
        if let StreamingProtocolCompat::OpenaiCompletions(compat) = compat {
            compat.requires_reasoning_content = Some(true);
        }
        let (thinking, effort) =
            thinking_params(self.variant.specs, &model.id, options.reasoning_level());
        let has_tools = !context.tools.is_empty();
        let params = model
            .sampling_params
            .get_or_insert_with(serde_json::Map::new);
        params.insert("thinking".into(), thinking);
        match effort {
            Some(effort) => params.insert("reasoning_effort".into(), json!(effort)),
            None => params.remove("reasoning_effort"),
        };
        // Streams tool arguments as they are generated; Pi sends it for every
        // tool request on these providers.
        if has_tools {
            params.insert("tool_stream".into(), json!(true));
        } else {
            params.remove("tool_stream");
        }
        let mut options = options.clone();
        if options.api_key.is_none() {
            options.api_key = self.api_key.clone();
        }
        let protocol = rpi_plugin_sdk::provider::completions(&model, &options, false);
        protocol.stream_simple(&model, context, &options).await
    }
}

fn manifest(variant: &Variant) -> serde_json::Value {
    json!({
        "id": variant.id, "models": models_for(variant),
        "apiKeyEnv": variant.env, "requiresAuth": true
    })
}

/// Registers both platforms. Each lives in its own module because the shared
/// registration macro defines fixed ABI function names.
pub(crate) fn register(api: &rpi_plugin_sdk::PluginApi) -> i32 {
    match global::register(api) {
        0 => china::register(api),
        failure => failure,
    }
}

mod global {
    use super::*;

    fn create(_: &Model, options: &SimpleStreamOptions) -> Arc<dyn Provider> {
        Arc::new(ZaiProvider::new(options.api_key.clone()))
    }

    crate::registration::register_provider!(
        GLOBAL.id,
        GLOBAL.base_url,
        "openai-completions",
        manifest(&GLOBAL),
        create
    );
}

mod china {
    use super::*;

    fn create(_: &Model, options: &SimpleStreamOptions) -> Arc<dyn Provider> {
        Arc::new(ZaiProvider::china(options.api_key.clone()))
    }

    crate::registration::register_provider!(
        CHINA.id,
        CHINA.base_url,
        "openai-completions",
        manifest(&CHINA),
        create
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_thinking_models_never_disable_and_use_documented_efforts() {
        for (level, effort) in [
            (ThinkingLevel::Off, "low"),
            (ThinkingLevel::Minimal, "low"),
            (ThinkingLevel::Low, "low"),
            (ThinkingLevel::Medium, "high"),
            (ThinkingLevel::High, "high"),
            (ThinkingLevel::Xhigh, "high"),
            (ThinkingLevel::Max, "max"),
        ] {
            let (thinking, got) = thinking_params(GLOBAL_SPECS, DEFAULT_MODEL_ID, level);
            assert_eq!(thinking["type"], "enabled");
            assert_eq!(thinking["clear_thinking"], false);
            assert_eq!(got, Some(effort));
        }
    }

    #[test]
    fn toggle_models_can_turn_thinking_off() {
        for id in ["glm-5.2", "glm-5-turbo", "glm-4.7"] {
            let (thinking, effort) = thinking_params(GLOBAL_SPECS, id, ThinkingLevel::Off);
            assert_eq!(thinking, json!({"type": "disabled"}));
            assert_eq!(effort, None);
        }
        let (thinking, effort) = thinking_params(CHINA_SPECS, "glm-4.6v", ThinkingLevel::Off);
        assert_eq!(thinking, json!({"type": "disabled"}));
        assert_eq!(effort, None);
    }

    #[test]
    fn effort_is_clamped_to_what_the_model_accepts() {
        // GLM-5.2 has no `low`, so low requests round up to `high`.
        let g = |id, level| thinking_params(GLOBAL_SPECS, id, level).1;
        assert_eq!(g("glm-5.2", ThinkingLevel::Low), Some("high"));
        assert_eq!(g("glm-5.2", ThinkingLevel::Max), Some("max"));
        // GLM-5 Turbo and GLM-4.7 take no effort field at all.
        assert_eq!(g("glm-4.7", ThinkingLevel::High), None);
        assert_eq!(g("glm-5-turbo", ThinkingLevel::Max), None);
    }

    #[test]
    fn global_catalog_lists_every_plan_model_on_the_global_endpoint() {
        let catalog = models_for(&GLOBAL);
        let ids: Vec<_> = catalog.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "glm-5.3",
                "glm-5.3-highspeed",
                "glm-5.3-flash",
                "glm-5.2",
                "glm-5.2-highspeed",
                "glm-5-turbo",
                "glm-4.7"
            ]
        );
        assert!(catalog.iter().all(|m| m.provider == "zai"
            && m.base_url == "https://api.z.ai/api/coding/paas/v4"
            && m.reasoning));
        assert_eq!(
            catalog
                .iter()
                .filter(|m| m.input.contains(&InputModality::Image))
                .count(),
            1
        );
    }

    #[test]
    fn china_catalog_uses_the_bigmodel_endpoint_and_its_own_key_variable() {
        let catalog = models_for(&CHINA);
        let ids: Vec<_> = catalog.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            ["glm-5.3", "glm-5.3-highspeed", "glm-5.3-flash", "glm-4.6v"]
        );
        assert!(catalog.iter().all(|m| m.provider == "zai-coding-cn"
            && m.base_url == "https://open.bigmodel.cn/api/coding/paas/v4"));
        assert_eq!(manifest(&CHINA)["apiKeyEnv"], "ZAI_CODING_CN_API_KEY");
        assert_eq!(manifest(&GLOBAL)["apiKeyEnv"], "ZAI_API_KEY");
        let vision = &catalog[3];
        assert_eq!(
            (vision.context_window, vision.max_tokens),
            (128_000, 16_384)
        );
        assert!(vision.input.contains(&InputModality::Image));
    }
}

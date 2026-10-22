//! Z.ai GLM Coding Plan policy belongs to this optional package. Wire parsing,
//! retries and the native streaming ABI use the maintained provider protocol.
//!
//! The catalog and request shape follow Pi (badlogic/pi-mono, `zai` provider):
//! every `zai-coding-plan` model on models.dev that supports tool calls, with
//! `thinking`, `reasoning_effort` and `tool_stream` set per model.
use rpi_ai::{
    model::{MaxTokensField, OpenaiCompletionsCompat, StreamingProtocolCompat},
    types::InputModality,
    Api, AssistantMessageEventStream, Context, Model, Provider, SimpleStreamOptions, ThinkingLevel,
};
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};

pub const PROVIDER_ID: &str = "zai";
pub const DEFAULT_MODEL_ID: &str = "glm-5.3";
pub const FLASH_MODEL_ID: &str = "glm-5.3-flash";
/// Coding Plan Chat Completions endpoint. The general API lives under
/// `/api/paas/v4` and does not consume plan quota; plan keys are separate.
const BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";

/// Thinking behaviour differs by generation. GLM-5.3 always thinks and accepts
/// only `reasoning_effort`; GLM-5.2 and earlier can switch thinking off.
struct Spec {
    id: &'static str,
    name: &'static str,
    image: bool,
    context: u64,
    /// Thinking can be turned off with `thinking.type = "disabled"`.
    toggle: bool,
    /// Accepted `reasoning_effort` values, lowest first. Empty means the model
    /// takes no effort parameter.
    efforts: &'static [&'static str],
}

const LOW_HIGH_MAX: &[&str] = &["low", "high", "max"];
const HIGH_MAX: &[&str] = &["high", "max"];

/// Order matters: the first entry is the default. Models marked deprecated on
/// models.dev stay listed because the Coding Plan still routes them.
const SPECS: &[Spec] = &[
    Spec {
        id: DEFAULT_MODEL_ID,
        name: "GLM-5.3",
        image: false,
        context: 1_000_000,
        toggle: false,
        efforts: LOW_HIGH_MAX,
    },
    Spec {
        id: "glm-5.3-highspeed",
        name: "GLM-5.3 Highspeed",
        image: false,
        context: 1_000_000,
        toggle: false,
        efforts: LOW_HIGH_MAX,
    },
    Spec {
        id: FLASH_MODEL_ID,
        name: "GLM-5.3 Flash",
        image: true,
        context: 1_000_000,
        toggle: false,
        efforts: LOW_HIGH_MAX,
    },
    Spec {
        id: "glm-5.2",
        name: "GLM-5.2 (deprecated)",
        image: false,
        context: 1_000_000,
        toggle: true,
        efforts: HIGH_MAX,
    },
    Spec {
        id: "glm-5.2-highspeed",
        name: "GLM-5.2 Highspeed (deprecated)",
        image: false,
        context: 1_000_000,
        toggle: true,
        efforts: HIGH_MAX,
    },
    Spec {
        id: "glm-5-turbo",
        name: "GLM-5 Turbo (deprecated)",
        image: false,
        context: 200_000,
        toggle: true,
        efforts: &[],
    },
    Spec {
        id: "glm-4.7",
        name: "GLM-4.7 (deprecated)",
        image: false,
        context: 204_800,
        toggle: true,
        efforts: &[],
    },
];

fn spec(id: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|spec| spec.id == id)
}

fn compatibility(supports_effort: bool) -> OpenaiCompletionsCompat {
    OpenaiCompletionsCompat {
        supports_store: Some(false),
        supports_developer_role: Some(false),
        supports_strict_mode: Some(false),
        supports_reasoning_effort: Some(supports_effort),
        // Tool turns must return the unmodified `reasoning_content`.
        requires_reasoning_content: Some(true),
        // Z.ai does not document whether hidden reasoning counts toward
        // `max_tokens`. Reserving the thinking budget on top of the answer
        // allowance only raises the requested ceiling (the models allow 128K),
        // and avoids an empty `finish_reason: "length"` answer if it does.
        reasoning_counts_toward_max_tokens: Some(true),
        max_tokens_field: Some(MaxTokensField::MaxTokens),
        ..Default::default()
    }
}

/// The thinking levels the selector (`/thinking`, Shift+Tab) offers for a
/// model. `None` hides a level; absent entries stay available. Only levels
/// that change what the server does are listed, so GLM-5.3 shows low / high /
/// max and no "off", and the models that only switch thinking on or off show
/// off / high.
fn level_map(spec: &Spec) -> BTreeMap<ThinkingLevel, Option<String>> {
    use ThinkingLevel::*;
    let has = |effort: &str| spec.efforts.iter().any(|candidate| *candidate == effort);
    let mut map = BTreeMap::from([(Minimal, None), (Medium, None)]);
    if !spec.toggle {
        map.insert(Off, None);
    }
    map.insert(Low, has("low").then(|| "low".to_string()));
    // Toggle-only models still need one "on" level, which reads as high.
    map.insert(High, Some("high".to_string()));
    map.insert(Xhigh, None);
    if has("max") {
        map.insert(Max, Some("max".to_string()));
    }
    map
}

pub fn models() -> Vec<Model> {
    SPECS
        .iter()
        .map(|spec| {
            let mut model = Model::new(
                spec.id,
                spec.name,
                Api::OpenaiCompletions,
                PROVIDER_ID,
                BASE_URL,
            );
            model.reasoning = true;
            if spec.image {
                model.input = vec![InputModality::Text, InputModality::Image];
            }
            model.context_window = spec.context;
            // The visible answer's allowance; the server accepts up to 128K.
            // This is a per-request default, not the maximum.
            model.max_tokens = 16_384;
            model.thinking_level_map = Some(level_map(spec));
            model.compat = Some(StreamingProtocolCompat::OpenaiCompletions(compatibility(
                !spec.efforts.is_empty(),
            )));
            // The Coding Plan bills in plan credits, not per token, so no cost
            // rates are declared. Users can add estimates in models.json.
            model
        })
        .collect()
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
    model_id: &str,
    level: ThinkingLevel,
) -> (serde_json::Value, Option<&'static str>) {
    let (toggle, efforts) = spec(model_id).map_or((false, LOW_HIGH_MAX), |s| (s.toggle, s.efforts));
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
    api_key: Option<String>,
    models: Vec<Model>,
}

impl ZaiProvider {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            models: models(),
        }
    }
}

#[async_trait::async_trait]
impl Provider for ZaiProvider {
    fn id(&self) -> &str {
        PROVIDER_ID
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
        model.provider = PROVIDER_ID.into();
        if model.base_url.is_empty() {
            model.base_url = BASE_URL.into();
        }
        let compat = model
            .compat
            .get_or_insert_with(|| StreamingProtocolCompat::OpenaiCompletions(compatibility(true)));
        if let StreamingProtocolCompat::OpenaiCompletions(compat) = compat {
            compat.requires_reasoning_content = Some(true);
        }
        let (thinking, effort) = thinking_params(&model.id, options.reasoning_level());
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
        // tool request on this provider.
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

fn create(_: &Model, options: &SimpleStreamOptions) -> Arc<dyn Provider> {
    Arc::new(ZaiProvider::new(options.api_key.clone()))
}

crate::registration::register_provider!(
    PROVIDER_ID,
    BASE_URL,
    "openai-completions",
    json!({
        "id": PROVIDER_ID, "models": models(),
        "apiKeyEnv": "ZAI_API_KEY", "requiresAuth": true
    }),
    create
);

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
            let (thinking, got) = thinking_params(DEFAULT_MODEL_ID, level);
            assert_eq!(thinking["type"], "enabled");
            assert_eq!(thinking["clear_thinking"], false);
            assert_eq!(got, Some(effort));
        }
    }

    #[test]
    fn selector_offers_only_levels_that_change_the_request() {
        use ThinkingLevel::*;
        let levels = |id: &str| {
            models()
                .into_iter()
                .find(|model| model.id == id)
                .unwrap()
                .supported_thinking_levels()
        };
        // Forced thinking: no off, three real efforts.
        for id in ["glm-5.3", "glm-5.3-highspeed", "glm-5.3-flash"] {
            assert_eq!(levels(id), [Low, High, Max]);
        }
        // Switchable thinking with high / max efforts.
        assert_eq!(levels("glm-5.2"), [Off, High, Max]);
        // Switchable thinking with no effort parameter.
        for id in ["glm-5-turbo", "glm-4.7"] {
            assert_eq!(levels(id), [Off, High]);
        }
    }

    #[test]
    fn toggle_models_can_turn_thinking_off() {
        for id in ["glm-5.2", "glm-5-turbo", "glm-4.7"] {
            let (thinking, effort) = thinking_params(id, ThinkingLevel::Off);
            assert_eq!(thinking, json!({"type": "disabled"}));
            assert_eq!(effort, None);
        }
    }

    #[test]
    fn effort_is_clamped_to_what_the_model_accepts() {
        // GLM-5.2 has no `low`, so low requests round up to `high`.
        assert_eq!(
            thinking_params("glm-5.2", ThinkingLevel::Low).1,
            Some("high")
        );
        assert_eq!(
            thinking_params("glm-5.2", ThinkingLevel::Max).1,
            Some("max")
        );
        // GLM-5 Turbo and GLM-4.7 take no effort field at all.
        assert_eq!(thinking_params("glm-4.7", ThinkingLevel::High).1, None);
        assert_eq!(thinking_params("glm-5-turbo", ThinkingLevel::Max).1, None);
    }

    #[test]
    fn catalog_lists_every_plan_model_on_the_coding_endpoint() {
        let catalog = models();
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
        assert!(catalog
            .iter()
            .all(|m| m.provider == PROVIDER_ID && m.base_url == BASE_URL && m.reasoning));
        assert!(catalog[2].input.contains(&InputModality::Image));
        assert_eq!(
            catalog
                .iter()
                .filter(|m| m.input.contains(&InputModality::Image))
                .count(),
            1
        );
    }
}

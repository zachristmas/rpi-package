//! Z.ai GLM Coding Plan policy belongs to this optional package. Wire parsing,
//! retries and the native streaming ABI use the maintained provider protocol.
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
        // allowance only raises the requested ceiling (the models allow 128K),
        // and avoids an empty `finish_reason: "length"` answer if it does.
        reasoning_counts_toward_max_tokens: Some(true),
        max_tokens_field: Some(MaxTokensField::MaxTokens),
        ..Default::default()
    }
}

pub fn models() -> Vec<Model> {
    [
        (DEFAULT_MODEL_ID, "GLM-5.3", vec![InputModality::Text]),
        (
            FLASH_MODEL_ID,
            "GLM-5.3 Flash",
            vec![InputModality::Text, InputModality::Image],
        ),
    ]
    .into_iter()
    .map(|(id, name, input)| {
        let mut model = Model::new(id, name, Api::OpenaiCompletions, PROVIDER_ID, BASE_URL);
        model.reasoning = true;
        model.input = input;
        model.context_window = 1_000_000;
        // The visible answer's allowance; the server accepts up to 128K. This is
        // a per-request default, not the maximum.
        model.max_tokens = 16_384;
        // Thinking is always on for GLM-5.3. `reasoning_effort` accepts
        // low / high / max, so every other level collapses onto those.
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

/// Maps the host thinking level to the only values GLM-5.3 accepts.
fn reasoning_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off | ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Max => "max",
        ThinkingLevel::Medium | ThinkingLevel::High | ThinkingLevel::Xhigh => "high",
    }
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
            .get_or_insert_with(|| StreamingProtocolCompat::OpenaiCompletions(compatibility()));
        if let StreamingProtocolCompat::OpenaiCompletions(compat) = compat {
            compat.requires_reasoning_content = Some(true);
        }
        let params = model
            .sampling_params
            .get_or_insert_with(serde_json::Map::new);
        // Reasoning cannot be disabled on GLM-5.3, and `"disabled"` is rejected,
        // so `--thinking off` sends the lowest effort instead of turning it off.
        params.insert("thinking".into(), json!({"type": "enabled"}));
        params.insert(
            "reasoning_effort".into(),
            json!(reasoning_effort(options.reasoning_level())),
        );
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
    fn effort_never_disables_thinking_and_uses_only_documented_values() {
        for (level, effort) in [
            (ThinkingLevel::Off, "low"),
            (ThinkingLevel::Minimal, "low"),
            (ThinkingLevel::Low, "low"),
            (ThinkingLevel::Medium, "high"),
            (ThinkingLevel::High, "high"),
            (ThinkingLevel::Xhigh, "high"),
            (ThinkingLevel::Max, "max"),
        ] {
            assert_eq!(reasoning_effort(level), effort);
        }
    }

    #[test]
    fn catalog_targets_the_coding_plan_endpoint() {
        let catalog = models();
        assert_eq!(catalog[0].id, DEFAULT_MODEL_ID);
        assert!(catalog
            .iter()
            .all(|m| m.provider == PROVIDER_ID && m.base_url == BASE_URL && m.reasoning));
        assert!(!catalog[0].input.contains(&InputModality::Image));
        assert!(catalog[1].input.contains(&InputModality::Image));
    }
}

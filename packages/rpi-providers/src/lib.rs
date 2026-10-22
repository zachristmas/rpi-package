//! One optional provider package, one native plugin entry, independent identities.
pub mod deepseek;
pub mod llama_cpp;
pub mod openrouter;
pub mod proxy;
mod registration;
pub mod zai;

rpi_plugin_sdk::export_plugin!(|api| {
    for register in [
        deepseek::register,
        openrouter::register,
        llama_cpp::register,
        proxy::register,
        zai::register,
    ] {
        let result = register(api);
        if result != 0 {
            return result;
        }
    }
    0
});

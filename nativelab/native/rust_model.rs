use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn input_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn into_c_string(value: &str) -> *mut c_char {
    CString::new(value).unwrap_or_else(|_| CString::new("").unwrap()).into_raw()
}

fn path_stem_lower(value: &str) -> String {
    let file = value
        .rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or(value);
    let stem = file.rsplit_once('.').map(|(left, _)| left).unwrap_or(file);
    stem.to_ascii_lowercase()
}

fn family_key(value: &str) -> &'static str {
    let name = path_stem_lower(value);
    let patterns: &[(&[&str], &str)] = &[
        (&["deepseek-r1"], "deepseek-r1"),
        (&["deepseek-coder", "deepseek_coder"], "deepseek-coder"),
        (&["deepseek"], "deepseek"),
        (&["mixtral"], "mixtral"),
        (&["mistral"], "mistral"),
        (&["llama-3", "llama3", "llama_3", "meta-llama-3"], "llama3"),
        (&["codellama", "code-llama", "code_llama"], "codellama"),
        (&["llama-2", "llama2", "llama_2"], "llama2"),
        (&["llama"], "llama2"),
        (&["phi-3", "phi3"], "phi3"),
        (&["phi"], "phi"),
        (&["qwen"], "qwen"),
        (&["gemma"], "gemma"),
        (&["yi-"], "yi"),
        (&["command-r", "command_r"], "command-r"),
        (&["orca", "openorca"], "orca"),
        (&["falcon"], "falcon"),
        (&["vicuna"], "vicuna"),
        (&["openchat"], "openchat"),
        (&["neural-chat", "neural_chat"], "neural-chat"),
        (&["starling"], "starling"),
        (&["zephyr"], "zephyr"),
        (&["solar"], "solar"),
    ];
    for (keywords, key) in patterns {
        if keywords.iter().any(|keyword| name.contains(keyword)) {
            return key;
        }
    }
    "default"
}

fn quant_type(value: &str) -> String {
    let file = value
        .rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or(value)
        .to_ascii_uppercase();
    let stem = file.strip_suffix(".GGUF").unwrap_or(&file).to_string();
    let patterns = [
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_XS", "IQ4_XS", "IQ4_NL",
        "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8",
        "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
        "IQ1_S", "IQ1_M", "IQ2_S", "IQ2_M", "IQ3_S", "IQ3_M",
        "Q2_K_S", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
        "BF16", "F16", "F32",
    ];
    for pattern in patterns {
        if stem.contains(pattern) {
            return pattern.to_string();
        }
    }
    "UNKNOWN".to_string()
}

#[no_mangle]
pub extern "C" fn nl_detect_family_key(filename: *const c_char) -> *mut c_char {
    into_c_string(family_key(&input_to_string(filename)))
}

#[no_mangle]
pub extern "C" fn nl_detect_quant_type(filename: *const c_char) -> *mut c_char {
    into_c_string(&quant_type(&input_to_string(filename)))
}

#[no_mangle]
pub extern "C" fn nl_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}

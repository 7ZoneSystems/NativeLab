pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        ((text.chars().count() + 3) / 4).max(1)
    }
}

pub fn json_object_span(text: &str) -> Option<(usize, usize)> {
    let mut start: Option<usize> = None;
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if start.is_none() {
            if ch == '{' {
                start = Some(idx);
                depth = 1;
            }
            continue;
        }

        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return start.map(|s| (s, idx + ch.len_utf8()));
                }
            }
            _ => {}
        }
    }
    None
}

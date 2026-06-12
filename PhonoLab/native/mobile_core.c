#include <stddef.h>
#include <stdint.h>

size_t phonolab_estimate_tokens(const char *text) {
    size_t bytes = 0;
    size_t words = 0;
    int in_word = 0;

    if (text == NULL) {
        return 0;
    }

    for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
        bytes += 1;
        if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            in_word = 0;
        } else if (!in_word) {
            words += 1;
            in_word = 1;
        }
    }

    size_t byte_cost = (bytes + 3) / 4;
    if (byte_cost < 1 && bytes > 0) {
        byte_cost = 1;
    }
    return byte_cost > words ? byte_cost : words;
}

int phonolab_context_fits(const char *text, int context_tokens, int reserved_output_tokens) {
    if (context_tokens <= 0) {
        return 0;
    }
    if (reserved_output_tokens < 0) {
        reserved_output_tokens = 0;
    }
    size_t used = phonolab_estimate_tokens(text);
    size_t limit = (size_t)context_tokens;
    size_t reserved = (size_t)reserved_output_tokens;
    if (reserved >= limit) {
        return 0;
    }
    return used <= (limit - reserved) ? 1 : 0;
}

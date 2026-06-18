package org.nativelab.phonolab.util

/**
 * Detects math expressions in chat messages for KaTeX rendering.
 * Matches both explicit LaTeX delimiters and common math patterns.
 *
 * NOTE: Android's ICU regex engine is stricter than Java's standard engine.
 * Use [{] and [}] for literal braces. Use lambda replacements to avoid group index issues.
 */
object MathHelper {

    // Explicit LaTeX delimiters — split into simple alternation
    private val DELIMITED_MATH = Regex(
        """\$\$[\s\S]+?\$\$|\$[^\$\n]+?\$|\\\(.*?\\\)|\\\[.*?\\\]"""
    )

    // \begin{equation}...\end{equation}
    private val ENV_BLOCK = Regex(
        """\\begin[{](equation|align|eqnarray)[*]?[}][\s\S]+?\\end[{](equation|align|eqnarray)[*]?[}]"""
    )

    // LaTeX commands that indicate math content
    private val LATEX_CMD = Regex(
        """\\(frac|sqrt|sin|cos|tan|sec|csc|cot|log|ln|exp|lim|sum|int|prod|partial|nabla|infty|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|phi|psi|rho|pi|tau|xi|zeta|eta|kappa|nu|varphi|varepsilon)\b"""
    )

    // Exponents: x^2, x^{n+1}, a^b
    private val EXPONENT = Regex("""[a-zA-Z0-9)\]]\^[\-]?\w+""")
    private val EXPONENT_BRACE = Regex("""[a-zA-Z0-9)\]]\^[{][^}]+[}]""")

    // Subscripts: a_n, x_{i}
    private val SUBSCRIPT = Regex("""[a-zA-Z]_\w+""")
    private val SUBSCRIPT_BRACE = Regex("""[a-zA-Z]_[{][^}]+[}]""")

    /** Returns true if the text contains math that should be rendered. */
    fun hasMath(text: String): Boolean {
        if (text.isEmpty()) return false
        return DELIMITED_MATH.containsMatchIn(text) ||
            ENV_BLOCK.containsMatchIn(text) ||
            LATEX_CMD.containsMatchIn(text) ||
            EXPONENT.containsMatchIn(text) ||
            EXPONENT_BRACE.containsMatchIn(text) ||
            SUBSCRIPT.containsMatchIn(text) ||
            SUBSCRIPT_BRACE.containsMatchIn(text)
    }

    /**
     * Pre-process text to wrap auto-detected math in $...$ for KaTeX.
     * Uses lambda replacements to avoid group index issues on Android ICU.
     */
    fun wrapAutoMath(text: String): String {
        // If already has explicit delimiters, don't touch it
        if (DELIMITED_MATH.containsMatchIn(text) || ENV_BLOCK.containsMatchIn(text)) {
            return text
        }

        var result = text

        // Wrap LaTeX commands with args: \frac{a}{b} → $\frac{a}{b}$
        // Match \cmd followed by optional {..} [..] groups
        result = Regex("""\\(frac|sqrt|sin|cos|tan|sec|csc|cot|log|ln|exp|lim|sum|int|prod)(\{[^}]*\}|\[[^\]]*\])*""").replace(result) { m ->
            val v = m.value
            // Only wrap if it looks like real math (has braces or is a known function)
            if (v.contains("{") || v.contains("[") || v.length > 5) "$" + v + "$" else v
        }

        // Wrap exponents: x^2, a^{n+1}
        result = EXPONENT_BRACE.replace(result) { m -> "$" + m.value + "$" }
        result = EXPONENT.replace(result) { m -> "$" + m.value + "$" }

        // Wrap subscripts: a_n, x_{i}
        result = SUBSCRIPT_BRACE.replace(result) { m -> "$" + m.value + "$" }
        result = SUBSCRIPT.replace(result) { m -> "$" + m.value + "$" }

        return result
    }
}

package org.nativelab.phonolab.util

object UiHelpers {
    fun calcPercent(done: Long, total: Long): Int =
        if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
}

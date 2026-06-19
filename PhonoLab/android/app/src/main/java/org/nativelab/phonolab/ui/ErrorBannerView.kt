package org.nativelab.phonolab.ui

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import org.nativelab.phonolab.R

/**
 * Red exclamation banner that appears top-right, auto-dismisses after timeout.
 * Non-fatal errors - the app keeps running, user sees a brief notification.
 */
class ErrorBannerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : FrameLayout(context, attrs, defStyleAttr) {

    private val textView: TextView
    private val dismissRunnable = Runnable { hide() }

    init {
        visibility = View.GONE
        elevation = 16f

        val container = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            val dp8 = dp(8)
            val dp12 = dp(12)
            setPadding(dp12, dp8, dp12, dp8)
            background = createErrorBackground()
        }

        val icon = ImageView(context).apply {
            setImageResource(android.R.drawable.ic_dialog_alert)
            setColorFilter(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(dp(18), dp(18)).apply {
                marginEnd = dp(8)
            }
        }

        textView = TextView(context).apply {
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            typeface = Typeface.DEFAULT_BOLD
            maxLines = 2
            ellipsize = android.text.TextUtils.TruncateAt.END
        }

        val closeBtn = ImageView(context).apply {
            setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
            setColorFilter(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(dp(20), dp(20)).apply {
                marginStart = dp(8)
            }
            setOnClickListener { hide() }
        }

        container.addView(icon)
        container.addView(textView, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        container.addView(closeBtn)

        addView(container, LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.TOP
        ))
    }

    fun show(message: String, durationMs: Long = 5000L) {
        textView.text = "\u26A0 $message"
        visibility = View.VISIBLE
        alpha = 0f
        animate().alpha(1f).setDuration(200).start()
        removeCallbacks(dismissRunnable)
        if (durationMs > 0) {
            postDelayed(dismissRunnable, durationMs)
        }
    }

    fun hide() {
        removeCallbacks(dismissRunnable)
        animate().alpha(0f).setDuration(200).withEndAction {
            visibility = View.GONE
        }.start()
    }

    private fun createErrorBackground(): android.graphics.drawable.GradientDrawable {
        return android.graphics.drawable.GradientDrawable().apply {
            setColor(Color.parseColor("#E0B91C1C"))
            cornerRadius = dp(8).toFloat()
        }
    }

    private fun dp(value: Int): Int {
        return (value * context.resources.displayMetrics.density).toInt()
    }
}

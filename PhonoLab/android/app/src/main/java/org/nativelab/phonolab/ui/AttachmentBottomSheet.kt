package org.nativelab.phonolab.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import org.nativelab.phonolab.R

class AttachmentBottomSheet : BottomSheetDialogFragment() {

    interface AttachmentCallback {
        fun onImageSelected()
        fun onDocumentSelected()
    }

    var callback: AttachmentCallback? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.bottom_sheet_attachment, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        view.findViewById<View>(R.id.btn_image).setOnClickListener {
            callback?.onImageSelected()
            dismiss()
        }

        view.findViewById<View>(R.id.btn_document).setOnClickListener {
            callback?.onDocumentSelected()
            dismiss()
        }
    }

    companion object {
        const val TAG = "AttachmentBottomSheet"
    }
}

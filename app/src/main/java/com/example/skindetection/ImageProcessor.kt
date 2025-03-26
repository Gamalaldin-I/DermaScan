package com.example.skindetection

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageProcessor {
    fun bitmapToByteBuffer(bitmap: Bitmap, imageSize: Int): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)

        // تأكد من حجم البافر المناسب لموديل (1, 224, 224, 3)
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3) // 4 Bytes لكل float
        byteBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {
                val pixel = scaledBitmap.getPixel(x, y)

                // استخراج القنوات اللونية وتطبيعها (قيم بين 0 و 1)
                byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // Red
                byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // Green
                byteBuffer.putFloat((pixel and 0xFF) / 255.0f)         // Blue
            }
        }
        return byteBuffer
    }
}

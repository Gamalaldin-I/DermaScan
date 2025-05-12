package com.example.skindetection

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteHelper(context: Context, modelPath: String) {
    private var interpreter: Interpreter

    init {
        val model = loadModelFile(context, modelPath)
        interpreter = Interpreter(model)
    }

    fun predict(inputBuffer: ByteBuffer): Int {
        val output = Array(1) { FloatArray(7) }  // تعديل الحجم إلى 7 بناءً على الموديل

        interpreter.run(inputBuffer, output)

        // استخراج أعلى قيمة (أكثر تصنيف محتمل)
        val predictedIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1

        return predictedIndex  // إرجاع التصنيف ذو الاحتمالية الأعلى
    }

    fun close() {
        interpreter.close()
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}

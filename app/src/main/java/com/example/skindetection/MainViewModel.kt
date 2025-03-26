package com.example.skindetection

import androidx.lifecycle.ViewModel

class MainViewModel : ViewModel() {
    fun result(result:Int):SkinDisease{
        val list = DataLap.list
        return list[result]
    }
}
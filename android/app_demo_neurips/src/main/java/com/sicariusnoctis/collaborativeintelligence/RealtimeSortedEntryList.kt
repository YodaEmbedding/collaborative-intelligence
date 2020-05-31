package com.sicariusnoctis.collaborativeintelligence

import com.github.mikephil.charting.data.Entry

/**
 * Sorted list data structure for realtime mostly-ordered data.
 *
 * Expects that items appended to list are nearly the largest element.
 */
class RealtimeSortedEntryList : ArrayList<Entry>() {
    private val unsupported = "Sorted list does not support this operation"

    override fun add(element: Entry): Boolean {
        val index = indexOfLast { it.x <= element.x }
        super.add(index + 1, element)
        return true
    }

    override fun addAll(elements: Collection<Entry>): Boolean {
        for (element in elements.sortedBy { it.x }) {
            add(element)
        }
        return true
    }

    override fun add(index: Int, element: Entry) = throw Exception(unsupported)
    override fun addAll(index: Int, elements: Collection<Entry>): Boolean =
        throw Exception(unsupported)

    override fun retainAll(elements: Collection<Entry>): Boolean = throw Exception(unsupported)
    override fun set(index: Int, element: Entry): Entry = throw Exception(unsupported)
}

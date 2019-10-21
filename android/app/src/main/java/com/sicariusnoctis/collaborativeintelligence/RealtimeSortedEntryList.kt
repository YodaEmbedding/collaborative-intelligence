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
    override fun addAll(index: Int, elements: Collection<Entry>): Boolean = throw Exception(unsupported)
    override fun retainAll(elements: Collection<Entry>): Boolean = throw Exception(unsupported)
    override fun set(index: Int, element: Entry): Entry = throw Exception(unsupported)
}

// /**
//  * Sorted list data structure for realtime mostly-ordered data.
//  *
//  * Expects that items appended to list are nearly the largest element.
//  */
// class RealtimeSortedEntryList : MutableList<Entry> {
//     private val list = mutableListOf<Entry>()
//     override val size get() = list.size
//     private val unsupported = "Sorted list does not support this operation"
//
//     // MutableList<T>
//
//     override fun add(element: Entry): Boolean {
//         val index = list.indexOfLast { it.x <= element.x }
//         list.add(index + 1, element)
//         return true
//     }
//
//     override fun addAll(elements: Collection<Entry>): Boolean {
//         for (element in elements.sortedBy { it.x }) {
//             add(element)
//         }
//         return true
//     }
//
//     override fun add(index: Int, element: Entry) = throw Exception(unsupported)
//     override fun addAll(index: Int, elements: Collection<Entry>): Boolean = throw Exception(unsupported)
//     override fun retainAll(elements: Collection<Entry>): Boolean = throw Exception(unsupported)
//     override fun set(index: Int, element: Entry): Entry = throw Exception(unsupported)
//
//     override fun clear() = list.clear()
//     override fun remove(element: Entry): Boolean = list.remove(element)
//     override fun removeAll(elements: Collection<Entry>): Boolean = list.removeAll(elements)
//     override fun removeAt(index: Int): Entry = list.removeAt(index)
//
//     // List<T>
//
//     override fun contains(element: Entry): Boolean = list.contains(element)
//     override fun containsAll(elements: Collection<Entry>): Boolean = list.containsAll(elements)
//     override fun get(index: Int): Entry = list[index]
//     override fun indexOf(element: Entry): Int = list.indexOf(element)
//     override fun isEmpty(): Boolean = list.isEmpty()
//     override fun iterator(): MutableIterator<Entry> = list.iterator()
//     override fun lastIndexOf(element: Entry): Int = list.lastIndexOf(element)
//     override fun listIterator(): MutableListIterator<Entry> = list.listIterator()
//     override fun listIterator(index: Int): MutableListIterator<Entry> = list.listIterator(index)
//     override fun subList(fromIndex: Int, toIndex: Int): MutableList<Entry> =
//         list.subList(fromIndex, toIndex)
// }

// /**
//  * Sorted list data structure for realtime mostly-ordered data.
//  *
//  * Expects that items appended to list are nearly the largest element.
//  */
// class RealtimeSortedList<T : Comparable<T>> : MutableList<T> {
//     private val list = mutableListOf<T>()
//     override val size get() = list.size
//     private val unsupported = "Sorted list does not support this operation"
//
//     // MutableList<T>
//
//     override fun add(element: T): Boolean {
//         list.add(list.indexOfLast { it <= element }, element)
//         return true
//     }
//
//     override fun addAll(elements: Collection<T>): Boolean {
//         for (element in elements.sorted()) {
//             add(element)
//         }
//         return true
//     }
//
//     override fun add(index: Int, element: T) = throw Exception(unsupported)
//     override fun addAll(index: Int, elements: Collection<T>): Boolean = throw Exception(unsupported)
//     override fun retainAll(elements: Collection<T>): Boolean = throw Exception(unsupported)
//     override fun set(index: Int, element: T): T = throw Exception(unsupported)
//
//     override fun clear() = list.clear()
//     override fun remove(element: T): Boolean = list.remove(element)
//     override fun removeAll(elements: Collection<T>): Boolean = list.removeAll(elements)
//     override fun removeAt(index: Int): T = list.removeAt(index)
//
//     // List<T>
//
//     override fun contains(element: T): Boolean = list.contains(element)
//     override fun containsAll(elements: Collection<T>): Boolean = list.containsAll(elements)
//     override fun get(index: Int): T = list[index]
//     override fun indexOf(element: T): Int = list.indexOf(element)
//     override fun isEmpty(): Boolean = list.isEmpty()
//     override fun iterator(): MutableIterator<T> = list.iterator()
//     override fun lastIndexOf(element: T): Int = list.lastIndexOf(element)
//     override fun listIterator(): MutableListIterator<T> = list.listIterator()
//     override fun listIterator(index: Int): MutableListIterator<T> = list.listIterator(index)
//     override fun subList(fromIndex: Int, toIndex: Int): MutableList<T> =
//         list.subList(fromIndex, toIndex)
// }
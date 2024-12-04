from typing import List, Any, Dict, Set, Generator

class StaticArray:
    def __init__(self, capacity: int):
        """
        Initialize a static array of a given capacity.
        """
        self.capacity = capacity
        self.array = [None] * capacity

    def set(self, index: int, value: int) -> None:
        """
        Set the value at a particular index.
        """
        if index < 0 or index >= self.capacity:
            raise IndexError("Index out of bounds.")
        self.array[index] = value
        

    def get(self, index: int) -> int:
        """
        Retrieve the value at a particular index.
        """
        if index < 0 or index >= self.capacity:
            raise IndexError("Index out of bounds.")
        return self.array[index]

class DynamicArray:
    def __init__(self):
        """
        Initialize an empty dynamic array.
        """
        self.array = []

    def append(self, value: int) -> None:
        """
        Add a value to the end of the dynamic array.
        """
        self.array.append(value)

    def insert(self, index: int, value: int) -> None:
        """
        Insert a value at a particular index.
        """
        if index < 0 or index > len(self.array):
            raise IndexError("Index out of bounds.")
        self.array.insert(index, value)

    def delete(self, index: int) -> None:
        """
        Delete the value at a particular index.
        """
        if index < 0 or index >= len(self.array):
            raise IndexError("Index out of bounds.")
        self.array.pop(index)

    def get(self, index: int) -> int:
        """
        Retrieve the value at a particular index.
        """
        if index < 0 or index >= len(self.array):
            raise IndexError("Index out of bounds.")
        return self.array[index]

class Node:
    def __init__(self, value: int):
        """
        Initialize a node.
        """
        self.value = value
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        """
        Initialize an empty singly linked list.
        """
        self.head = None

    def append(self, value: int) -> None:
        """
        Add a node with a value to the end of the linked list.
        """
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            return
        tail = self.get_tail()
        tail.next = new_node

    def insert(self, position: int, value: int) -> None:
        """
        Insert a node with a value at a particular position.
        """
        new_node = Node(value)
        if position == 0:
            new_node.next = self.head
            self.head = new_node
            return

        current = self.head
        count = 0
        while current and count < position - 1:
            current = current.next
            count += 1

        if not current:
            raise IndexError("Position out of bounds.")
        
        new_node.next = current.next
        current.next = new_node

    def delete(self, value: int) -> None:
        """
        Delete the first node with a specific value.
        """
        if not self.head:
            return
        if self.head.value == value:
            self.head = self.head.next
            return

        current = self.head
        while current.next and current.next.value != value:
            current = current.next

        if current.next:
            current.next = current.next.next

    def find(self, value: int) -> Node:
        """
        Find a node with a specific value.
        """
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def size(self) -> int:
        """
        Returns the number of elements in the linked list.
        """
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def is_empty(self) -> bool:
        """
        Checks if the linked list is empty.
        """
        return self.head is None

    def print_list(self) -> None:
        """
        Prints all elements in the linked list.
        """
        current = self.head
        while current:
            print(current.value, end=" -> ")
            current = current.next
        print("None")
    
    def reverse(self) -> None:
        """
        Reverse the linked list in-place.
        """
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
    
    def get_head(self) -> Node:
        """
        Returns the head node of the linked list.
        """
        return self.head
    
    def get_tail(self) -> Node:
        """
        Returns the tail node of the linked list.
        """
        current = self.head
        while current and current.next:
            current = current.next
        return current

class DoubleNode:
    def __init__(self, value: int, next_node = None, prev_node = None):
        """
        Initialize a double node with value, next, and previous.
        """
        self.value = value
        self.next = next_node
        self.prev = prev_node

class DoublyLinkedList:
    def __init__(self):
        """
        Initialize an empty doubly linked list.
        """
        self.head = None
        self.tail = None

    def append(self, value: int) -> None:
        """
        Add a node with a value to the end of the linked list.
        """
        new_node = DoubleNode(value)
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def insert(self, position: int, value: int) -> None:
        """
        Insert a node with a value at a particular position.
        """
        new_node = DoubleNode(value)
        if position == 0:
            if not self.head:
                self.head = self.tail = new_node
            else:
                new_node.next = self.head
                self.head.prev = new_node
                self.head = new_node
            return

        current = self.head
        count = 0
        while current and count < position - 1:
            current = current.next
            count += 1

        if not current:
            raise IndexError("Position out of bounds.")
        
        new_node.next = current.next
        new_node.prev = current
        if current.next:
            current.next.prev = new_node
        current.next = new_node
        if new_node.next is None:
            self.tail = new_node

    def delete(self, value: int) -> None:
        """
        Delete the first node with a specific value.
        """
        current = self.head
        while current:
            if current.value == value:
                if current.prev:
                    current.prev.next = current.next
                if current.next:
                    current.next.prev = current.prev
                if current == self.head:
                    self.head = current.next
                if current == self.tail:
                    self.tail = current.prev
                return
            current = current.next
        raise ValueError(f"Value {value} not found in the list.")

    def find(self, value: int) -> DoubleNode:
        """
        Find a node with a specific value.
        """
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def size(self) -> int:
        """
        Returns the number of elements in the linked list.
        """
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def is_empty(self) -> bool:
        """
        Checks if the linked list is empty.
        """
        return self.head is None

    def print_list(self) -> None:
        """
        Prints all elements in the linked list.
        """
        current = self.head
        while current:
            print(current.value, end=" <-> ")
            current = current.next
        print("None")
    
    def reverse(self) -> None:
        """
        Reverse the linked list in-place.
        """
        current = self.head
        prev_node = None
        while current:
            next_node = current.next
            current.next = prev_node
            current.prev = next_node
            prev_node = current
            current = next_node
        self.head, self.tail = self.tail, self.head
    
    def get_head(self) -> DoubleNode:
        """
        Returns the head node of the linked list.
        """
        return self.head
    
    def get_tail(self) -> DoubleNode:
        """
        Returns the tail node of the linked list.
        """
        return self.tail

class Queue:
    def __init__(self):
        """
        Initialize an empty queue.
        """
        self.queue = []

    def enqueue(self, value: int) -> None:
        """
        Add a value to the end of the queue.
        """
        self.queue.append(value)

    def dequeue(self) -> int:
        """
        Remove a value from the front of the queue and return it.
        """
        if self.is_empty():
            raise IndexError("Queue is empty.")
        return self.queue.pop(0)

    def peek(self) -> int:
        """
        Peek at the value at the front of the queue without removing it.
        """
        if self.is_empty():
            raise IndexError("Queue is empty.")
        return self.queue[0]

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        """
        return len(self.queue) == 0

class TreeNode:
    def __init__(self, value: int):
        """
        Initialize a tree node with value.
        """
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    class TreeNode:
        def __init__(self, value: int):
            """
            Initialize a tree node with value.
            """
            self.value = value
            self.left = None
            self.right = None
    
    
    def __init__(self):
        """
        Initialize an empty binary search tree.
        """
        self.root = None

    def insert(self, value: int) -> None:
        """
        Insert a node with a specific value into the binary search tree.
        """
        def _insert(node, value):
            if not node:
                return self.TreeNode(value)
            if value < node.value:
                node.left = _insert(node.left, value)
            else:
                node.right = _insert(node.right, value)
            return node

        self.root = _insert(self.root, value)

    def delete(self, value: int) -> None:
        """
        Remove a node with a specific value from the binary search tree.
        """
        def _delete(node, value):
            if not node:
                return None
            if value < node.value:
                node.left = _delete(node.left, value)
            elif value > node.value:
                node.right = _delete(node.right, value)
            else:
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                min_larger_node = self._find_min(node.right)
                node.value = min_larger_node.value
                node.right = _delete(node.right, min_larger_node.value)
            return node

        self.root = _delete(self.root, value)

    def search(self, value: int) -> TreeNode:
        """
        Search for a node with a specific value in the binary search tree.
        """
        def _search(node, value):
            if not node or node.value == value:
                return node
            if value < node.value:
                return _search(node.left, value)
            return _search(node.right, value)

        return _search(self.root, value)
        

    def inorder_traversal(self) -> List[int]:
        """
        Perform an in-order traversal of the binary search tree.
        """
        result = []

        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.value)
                _inorder(node.right)

        _inorder(self.root)
        return result
    
    def size(self) -> int:
        """
        Returns the number of nodes in the tree.
        """
        def _size(node):
            if not node:
                return 0
            return 1 + _size(node.left) + _size(node.right)

        return _size(self.root)

    def is_empty(self) -> bool:
        """
        Checks if the tree is empty.
        """
        return self.root is None

    def height(self) -> int:
        """
        Returns the height of the tree.
        """
        def _height(node):
            if not node:
                return -1
            return 1 + max(_height(node.left), _height(node.right))

        return _height(self.root)

    def preorder_traversal(self) -> List[int]:
        """
        Perform a pre-order traversal of the tree.
        """
        result = []

        def _preorder(node):
            if node:
                result.append(node.value)
                _preorder(node.left)
                _preorder(node.right)

        _preorder(self.root)
        return result

    def postorder_traversal(self) -> List[int]:
        """
        Perform a post-order traversal of the tree.
        """
        result = []

        def _postorder(node):
            if node:
                _postorder(node.left)
                _postorder(node.right)
                result.append(node.value)

        _postorder(self.root)
        return result

    def level_order_traversal(self) -> List[int]:
        """
        Perform a level order (breadth-first) traversal of the tree.
        """
        result = []
        if not self.root:
            return result

        queue = [self.root]  
        while queue:
            node = queue.pop(0)
            result.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    def _find_min(self, node):
        while node and node.left:
            node = node.left
        return node

    def _find_max(self, node):
        while node and node.right:
            node = node.right
        return node
    
    def minimum(self) -> TreeNode:
        """
        Returns the node with the minimum value in the tree.
        """
        return self._find_min(self.root)

    def maximum(self) -> TreeNode:
        """
        Returns the node with the maximum value in the tree.
        """
        return self._find_max(self.root)
    
    def is_valid_bst(self) -> bool:
        """
        Check if the tree is a valid binary search tree.
        """
        def _is_valid(node, low, high):
            if not node:
                return True
            if not (low < node.value < high):
                return False
            return _is_valid(node.left, low, node.value) and _is_valid(node.right, node.value, high)

        return _is_valid(self.root, float('-inf'), float('inf'))

def insertion_sort(lst: List[int]) -> List[int]:
    for i in range(1, len(lst)):
        key = lst[i]
        j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key
    return lst

def selection_sort(lst: List[int]) -> List[int]:
    for i in range(len(lst)):
        min_idx = i
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[min_idx]:
                min_idx = j
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
    return lst

def bubble_sort(lst: List[int]) -> List[int]:
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst

def shell_sort(lst: List[int]) -> List[int]:
    gap = len(lst) // 2
    while gap > 0:
        for i in range(gap, len(lst)):
            temp = lst[i]
            j = i
            while j >= gap and lst[j - gap] > temp:
                lst[j] = lst[j - gap]
                j -= gap
            lst[j] = temp
        gap //= 2
    return lst

def merge_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(lst: List[int]) -> List[int]:
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    left = [x for x in lst if x < pivot]
    middle = [x for x in lst if x == pivot]
    right = [x for x in lst if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

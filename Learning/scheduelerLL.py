#schedualing system for advising meetings in cs program
#each meeting should be scheduled when a student has completed 50% of their academic program
#each coruse has at most one prereq taken before
#there is only one path through the program
#take all the inputs through their sequence and returns the name of the course they will be taking when they are halfwat through their sequence of courses

#scheduling system implementation using a linkedlist

#steps initalize linkedlist
#add first set
#check if 
import sys

class Node:
    def __init__(self, data):
        self.item = data
        self.ref = None

class LinkedList:
    def __init__(self):
        self.start_node = None

    def traverse_list(self):
        if self.start_node is None:
            #print("List has no element")
            return
        else:
            n = self.start_node
            while n is not None:
                #print(n.item , " ")
                n = n.ref
    
    def insert_at_start(self, data):
        new_node = Node(data)
        new_node.ref = self.start_node
        self.start_node= new_node

    def insert_at_end(self, data):
        new_node = Node(data)
        if self.start_node is None:
            self.start_node = new_node
            return
        n = self.start_node
        while n.ref is not None:
            n= n.ref
        n.ref = new_node;
    
    def insert_after_item(self, x, data):

        n = self.start_node
        #print(n.ref)
        while n is not None:
            if n.item == x:
                break
            n = n.ref
        if n is None:
            #print("item not in the list")
            pass
        else:
            new_node = Node(data)
            new_node.ref = n.ref
            n.ref = new_node
    
    def insert_before_item(self, x, data):
        if self.start_node is None:
            print("List has no element")
            return

        if x == self.start_node.item:
            new_node = Node(data)
            new_node.ref = self.start_node
            self.start_node = new_node
            return

        n = self.start_node
        #print(n.ref)
        while n.ref is not None:
            if n.ref.item == x:
                break
            n = n.ref
        if n.ref is None:
            print("item not in the list")
        else:
            new_node = Node(data)
            new_node.ref = n.ref
            n.ref = new_node

    def search_item(self, x):
        if self.start_node is None:
            #print("List has no elements")
            return
        n = self.start_node
        while n is not None:
            if n.item == x:
                #print("Item found")
                return True
            n = n.ref
        #print("item not found")
        return False
    
    def get_count(self):
        if self.start_node is None:
            return 0;
        n = self.start_node
        count = 0;
        while n is not None:
            count = count + 1
            n = n.ref
        return count

    def traverse_list(self):
      if self.start_node is None:
          print("List has no element")
          return
      else:
          n = self.start_node
          while n is not None:
              print(n.item , " ")
              n = n.ref

    def get_head(self):
        return self.start_node.item
      
    
      
     

def findTail(pairs):
    elem1 = []
    elem2 = []
    for pair in pairs:
        elem1.append(pair[0])
        elem2.append(pair[1])
    for elem in elem2:
        if elem not in elem1:
            tail = elem
    return tail

def findindexList2(item, pairs):
    elem1 = []
    elem2 = []
    for pair in pairs:
        elem1.append(pair[0])
        elem2.append(pair[1])
    for elem in elem2:
        if elem not in elem1:
            tail = elem
    return elem2.index(item)

def findindexList1(item, pairs):
    elem1 = []
    elem2 = []
    for pair in pairs:
        elem1.append(pair[0])
        elem2.append(pair[1])
    for elem in elem2:
        if elem not in elem1:
            tail = elem
    return elem1.index(item)



def findMidpointCourse(pairs):
  # IMPLEMENTATION GOES HERE
  classesList = LinkedList()
  tail = findTail(pairs)
  classesList.insert_at_end(tail)
  pairs2 = dict(pairs)
  for pair in pairs2:
      if pair[1] == tail:
          classesList.insert_at_end(pair[0])
      else:
          head = classesList.get_head()
          index = findindexList2(head, pairs)
          classesList.insert_at_start(pairs[index][0])
  return classesList



# DO NOT MODIFY BELOW THIS LINE
def main():
  pairs = []

  for line in sys.stdin:
    if len(line.strip()) == 0:
      continue

    line = line.rstrip()

    pairs.append(line.split(" "))

  print(findMidpointCourse(pairs))

main()
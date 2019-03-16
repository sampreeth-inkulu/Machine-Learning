import math
import operator
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

def get_impurity(c1, c2):																# To obtain impurity of a node
	total = c1 + c2
	if(total == 0):
		return 0
	p1 = c1/total
	p2 = c2/total

	impurity = 0
	if(p1 != 0):
		impurity -= p1*math.log2(p1)
	if(p2 != 0):
		impurity -= p2*math.log2(p2)

	return impurity

def find_gain(c11, c12, c21, c22, impurity):											# Finds gain for an attribute test
	c1 = c11 + c12
	c2 = c21 + c22
	total = c1 + c2

	if(total == 0):
		return 0

	return impurity - c1*get_impurity(c11, c12)/total - c2*get_impurity(c21, c22)/total

class Node:
	"""docstring for Node"""
	def __init__(self):
		self.left = None																# Yes answer
		self.right = None																# No answer
		self.test_word = -1
		self.c1 = 0
		self.c2 = 0
		self.impurity = 1

	def predict_class(self, instance):													# Predicts class of an instance recursively
		if(self.left == None and self.right == None):									# Leaf node?
			if(self.c1 > self.c2):
				return 1
			else:
				return 2

		if(instance[self.test_word] == 1):
			return self.left.predict_class(instance)
		else:
			return self.right.predict_class(instance)

	def build_decision_tree(self, data, num_words, max_depth):							# Builds binary Decision tree
		for doc in data:
			self.c2 += doc[-1] - 1
		self.c1 = len(data) - self.c2
		
		if(max_depth == 0):
			return self

		self.impurity = get_impurity(self.c1, self.c2)
		if(self.impurity == 0):
			return self

		max_gain, best_word = -1, -1
		for word in range(num_words):													# Choose best attribute
			c11, c12, c21, c22 = 0, 0, 0, 0												# Counting c11, c12, c21, c22
			for doc in data:
				if(doc[word] == 1):
					if(doc[-1] == 1):
						c11 += 1
					else:
						c12 += 1
				else:
					if(doc[-1] == 1):
						c21 += 1
					else:
						c22 += 1

			gain = find_gain(c11, c12, c21, c22, self.impurity)							# Find gain for this test
			if(gain > max_gain):
				max_gain = gain
				best_word = word

		self.test_word = best_word
		data1 = []
		data2 = []
		for doc in data:																# Dividing data for children
			if(doc[best_word] == 1):
				data1.append(doc)
			else:
				data2.append(doc)

		self.left = Node()																# Build tree recursively
		self.left.build_decision_tree(data1, num_words, max_depth - 1)
		self.right = Node()
		self.right.build_decision_tree(data2, num_words, max_depth - 1)
		return self

	def print_tree(self, level = 0):													# Debugging utility
		print("\t"*level, self.test_word, self.c1, self.c2)
		if(self.left != None):
			self.left.print_tree(level + 1)
		if(self.right != None):
			self.right.print_tree(level + 1)

if __name__ == '__main__':

	docs = []
	with open("words.txt", "r") as words_file:											# Counting num_words
		for num_words, t in enumerate(words_file, 1):
			pass

	train_data = []																		# Organising train data into a list
	with open("trainlabel.txt", "r") as labels_file:
		for line in labels_file:
			temp = [0]*num_words
			temp.append(int(line))
			train_data.append(temp)
	
	with open("traindata.txt","r") as data_file:
		for line in data_file:
			temp = list(map(int, line.split())) 
			train_data[temp[0] - 1][temp[1] - 1] = 1

	test_data = []																		# Organising test data into a list
	with open("testlabel.txt", "r") as labels_file:
		for line in labels_file:
			temp = [0]*num_words
			temp.append(int(line))
			test_data.append(temp)
	
	with open("testdata.txt","r") as data_file:
		for line in data_file:
			temp = list(map(int, line.split())) 
			test_data[temp[0] - 1][temp[1] - 1] = 1

	# Got the matrix data!

	train_accuracy = 0
	train_acc = []
	test_acc = []
	max_depth = 0
	while train_accuracy != 1:															# Train until 100% train accuracy by increasing max_depth
		print("\nMaximum depth:", max_depth)
		root = Node()
		root.build_decision_tree(train_data, num_words, max_depth)						# Build tree
		max_depth += 1
		# root.print_tree()
		train_accuracy = 0																# Finding train accuracy
		for instance in train_data:
			if(instance[-1] == root.predict_class(instance[:-1])):
				train_accuracy += 1
		train_accuracy /= len(train_data)
		train_acc.append(train_accuracy*100)
		print("Train accuracy", train_acc[-1], "%")
		
		test_accuracy = 0																# Finding test accuracy
		for instance in test_data:
			if(instance[-1] == root.predict_class(instance[:-1])):
				test_accuracy += 1
		test_accuracy /= len(test_data)
		test_acc.append(test_accuracy*100)
		print("Test accuracy", test_acc[-1], "%")

	index, value = max( enumerate(test_acc), key = operator.itemgetter(1))				# Max test accuracy for?
	print("Maximum test accuracy =", value, "%  obtained for depth =",index)

	print("\nscikit learn decision tree algorithm")
	train_accuracy = 0
	depth_max = 1
	while train_accuracy != 1:															# scikit learn classifier
		print("\nMaximum depth:", depth_max)
		clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth_max)
		clf.fit( [instance[:-1] for instance in train_data], [instance[-1] for instance in train_data] )
		train_accuracy = clf.score( [instance[:-1] for instance in train_data], [instance[-1] for instance in train_data] )
		print("Test accuracy", 100*clf.score( [instance[:-1] for instance in test_data], [instance[-1] for instance in test_data] ))
		depth_max += 1

	depth_axis = range(max_depth)														# Plotting graph
	plt.plot(depth_axis, train_acc, marker = 'o', color = 'b', label = 'train accuracy')
	plt.plot(depth_axis, test_acc, marker = 'o', color = 'g', label = 'test accuracy')
	plt.title("Train & Test accuracies vs Max Depth")
	plt.xlabel("Maximum Depth allowed")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()
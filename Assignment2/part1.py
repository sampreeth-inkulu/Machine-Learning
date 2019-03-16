import math
import xlrd																	# For reading excel file
from sklearn.tree import DecisionTreeClassifier

def get_impurity(c0, c1, measure="gini"):									# Returns impurity of a node w.r.to measure
	total = c0 + c1
	if(total == 0):
		return 0
	p0 = c0/total
	p1 = c1/total

	if(measure == "gini"):
		impurity = 1 - p0**2 - p1**2
	elif(measure == "information gain"):
		impurity = 0
		if(p0 != 0):
			impurity -= p0*math.log2(p0)
		if(p1 != 0):
			impurity -= p1*math.log2(p1)

	return impurity


class Node:																	# Class Node
	"""docstring for Node"""
	def __init__(self, examples = None):
		self.child1 = None
		self.child2 = None
		self.child3 = None

		self.examples = examples
		self.c1 = [ row[-1] for row in examples ].count("yes")
		self.total = len(examples)
		self.c0 = self.total -self.c1
		self.impurity = 1

		self.attribute_test = -1
		self.first_value = None
		self.second_value = None
		self.third_value = None

	def set_impurity(self, measure = "gini"):								# Set impurity of a node
		self.impurity = get_impurity(self.c0, self.c1, measure)
		return self.impurity

	def predict_class(self, instance):										# Predicts class of an example recursively
		if(self.child1 == None and self.child2 == None and self.child3 == None):
			if(self.c0 == 0):
				return "yes"
			else:
				return "no"
		value = instance[self.attribute_test]
		if(self.child1 != None and self.first_value == value):
			return self.child1.predict_class(instance)
		if(self.child2 != None and self.second_value == value):
			return self.child2.predict_class(instance)
		if(self.child3 != None and self.third_value == value):
			return self.child3.predict_class(instance)

	def print_tree(self, attribute_names, attribute_values, level = -1):	# Prints tree recursively
		if(self.child1 == None and self.child2 == None and self.child3 == None):	# Leaf Node?
			if(self.c0 == 0):
				print(" : yes", end = "")
			else:
				print(" : no", end = "")
		else:
			if(self.child1 != None):
				if(level != -1):
					print("\n", "\t"*level, "|", end = " ")
				else:
					print()
				print(attribute_names[self.attribute_test], "=", self.first_value, end = "")
				self.child1.print_tree(attribute_names, attribute_values, level + 1)

			if(self.child2 != None):
				if(level != -1):
					print("\n", "\t"*level, "|", end = " ")
				else:
					print()
				print(attribute_names[self.attribute_test], "=", self.second_value, end = "")
				self.child2.print_tree(attribute_names, attribute_values, level + 1)

			if(self.child3 != None):
				if(level != -1):
					print("\n", "\t"*level, "|", end = " ")
				else:
					print()
				print(attribute_names[self.attribute_test], "=", self.third_value, end = "")
				self.child3.print_tree(attribute_names, attribute_values, level + 1)

def find_gain(temp, attribute_values, impurity, measure):				# Finds gain for an attribute test at a node

	temp0 = []
	temp1 = []
	temp2 = []
	for row in temp:
		if(row[0] == attribute_values[0]):
			temp0.append(row[-1])
		elif(row[0] == attribute_values[1]):
			temp1.append(row[-1])
		elif(len(attribute_values) > 2 and row[0] == attribute_values[2]):
			temp2.append(row[-1])

	c00 = temp0.count("no")												# Counting c00, c01,   c10, c11
	c01 = len(temp0) - c00
	c10 = temp1.count("no")
	c11 = len(temp1) - c10
	total = c00 + c01 + c10 + c11
	if(len(attribute_values) > 2):
		c20 = temp2.count("no")
		c21 = len(temp2) - c20
		total += c20 + c21
		return impurity - (c00+c01)*get_impurity(c00, c01, measure)/total - (c10+c11)*get_impurity(c10, c11, measure)/total - (c20+c21)*get_impurity(c20, c21, measure)/total
	return impurity - (c00+c01)*get_impurity(c00, c01, measure)/total - (c10+c11)*get_impurity(c10, c11, measure)/total

# Multi way split of attributes
def build_decision_tree(root, data, attribute_values, measure = "gini"):

	if(root == None):
		return root
	max_gain = -1
	col = -1
	root.set_impurity(measure)
	if(root.impurity == 0):															# Most homogeneous node?
		return root

	for i in range(len(data[0]) - 1):												# Which attribute test to choose?
		temp = [ [row[i], row[-1]] for row in data ]
		gain = find_gain(temp, attribute_values[i], root.impurity, measure)			# Find gain for this test
		if(gain > max_gain):
			max_gain = gain
			col = i

	root.attribute_test = col														# Set test variables of the node
	root.first_value = attribute_values[col][0]
	root.second_value = attribute_values[col][1]
	if(len(attribute_values[col]) > 2):
		root.third_value = attribute_values[col][2]
	data1, data2, data3 = [], [], []
	for row in data:																# Divide data among children
		if(row[col] == attribute_values[col][0]):
			data1.append(row)
		elif(row[col] == attribute_values[col][1]):
			data2.append(row)
		elif(len(attribute_values) > 2 and row[col] == attribute_values[col][2]):
			data3.append(row)
	if(len(data1)):																	# Set children links and build decision tree recursively
		child = Node(data1)
		root.child1 = child
		build_decision_tree(root.child1, data1, attribute_values, measure)
	if(len(data2)):
		child = Node(data2)
		root.child2 = child
		build_decision_tree(root.child2, data2, attribute_values, measure)
	if(len(data3)):
		child = Node(data3)
		root.child3 = child
		build_decision_tree(root.child3, data3, attribute_values, measure)
	return root

def func(str):																		# Used to map values to integers
	if(str == "low"):
		return 0
	elif(str == "med"):
		return 1
	elif(str == "high"):
		return 2
	elif(str == "yes"):
		return 1
	elif(str == "no"):
		return 0
	else:
		return str

def inv_func(label):																# Used to map integers to values
	if(label == 0):
		return "no"
	elif(label == 1):
		return "yes"

if __name__ == '__main__':
	
	wb = xlrd.open_workbook("dataset for part 1.xlsx")								# File name
	sheet0 = wb.sheet_by_index(0)
	sheet1 = wb.sheet_by_index(1)

	train_data = []																	# Storing train data in a list
	for i in range(sheet0.nrows):
		temp = []
		for j in range(sheet0.ncols):
			temp.append(sheet0.cell_value(i, j))
		train_data.append(temp)

	test_data = []																	# Storing test data in a list
	for i in range(sheet1.nrows):
		temp = []
		for j in range(sheet1.ncols):
			temp.append(sheet1.cell_value(i, j))
		test_data.append(temp)

	attribute_names = train_data[0][0:-1]											# Saving names of attributes to help while printing
	attribute_values = []
	for i in range(len(attribute_names)):
		attribute_values.append( list(set([ row[i] for row in train_data[1:] ])) )

	train_data = train_data[1:]
	test_data = test_data[1:]

	impurity = []
	for measure in ["gini", "information gain"]:									# Using the built model, for both measures
		print("\nWith impurity measure as",measure)
		root = Node(train_data)
		root = build_decision_tree(root, train_data, attribute_values, measure)
		root.print_tree(attribute_names, attribute_values)
		impurity.append(root.impurity)
		print("\n\nRoot node", measure, "=", root.impurity)

		accuracy = 0
		print("Predictions on test data")
		for instance in test_data:
			prediction = root.predict_class(instance[:-1])
			print(prediction)
			if(prediction == instance[-1]):
				accuracy += 1
		print("Accuracy = ", 100*accuracy/len(test_data), "%")

	print("\nsklearn DecisionTreeClassifier")
	for measure in ["gini", "entropy"]:												# Using scikit learn classifier, for both measures
		print("\nWith impurity measure as ", measure, "\n")
		clf = DecisionTreeClassifier(criterion = measure)
		clf.fit([list(map(func, instance[:-1])) for instance in train_data], [func(instance[-1]) for instance in train_data])
		X = [list(map(func, instance[:-1])) for instance in test_data]
		y = [func(instance[-1]) for instance in test_data]
		print("Predictions", *list(map(inv_func, clf.predict(X))), sep = "\n" )
		print("Accuracy = ", 100*clf.score(X,y),"%")

	print("Root node impurity is", impurity[0], "for gini and is", impurity[1], "for entropy")

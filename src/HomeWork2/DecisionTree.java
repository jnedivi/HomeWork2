package HomeWork2;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import java.util.LinkedList;
import java.util.Queue;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	
	Instances instances; // all the instances that have reaches this node
	int typeOneInstances; // red dots
	int typeTwoInstances; // blue dots
	int numberOfInstances;
	
	//TODO printNode!!	
	public void printNode(String tab) {
	        
		StringBuilder str = new StringBuilder();
	        if (parent == null) {
	            str.append("Root\n");
	        } else {
	            int attributeValue = (int) instances.get(0).value(parent.attributeIndex);
	            str.append(tab + "if attribute " + attributeIndex + " = " + attributeValue + "\n");
	        }
	        str.append(tab);
	        if (children == null) {
	            str.append("Leaf. ");
	        }
	        str.append("Returning value: " + returnValue);
	        System.out.println(str.toString());
	        if (children != null) {
	            tab += "\t";
	            for (int i = 0; i < children.length; i++) {
	                Node child = children[i];
	                if (child != null) {
	                    child.printNode(tab);
	                }
	            }
	        }
	    
		}
}

public class DecisionTree implements Classifier {
	private Node rootNode;
	
	public enum ImpurityMeasure {
		Entropy, Gini, Null
	};
	
	private ImpurityMeasure m_ImpurityMeasure = ImpurityMeasure.Null;
	
	public double[] k_p_values = {0.005, 0.05, 0.25, 0.5, 0.75, 1};
	
	private double m_p_value = 1;

	
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		this.buildTree(arg0);
	}
    
	public void buildTree(Instances dataSet) throws Exception {
		this.rootNode = new Node();
		Queue<Node> tree = new LinkedList<Node>();
		tree.add(this.rootNode);
		this.rootNode.instances = new Instances(dataSet);
		updateInstancesForNode(this.rootNode);
		
		while(!tree.isEmpty()){
			Node current = tree.poll();
			int bestAttributeIndex = getBestAttribute(current, dataSet);
			
			if(bestAttributeIndex != -1){
				createChildren(current, bestAttributeIndex, dataSet);

				int df = getDf(current);
				
				//System.out.println(calcChiSquare(current));
				// System.out.println(getChiSquareValue(df));
				if(m_p_value != 1){
					if(calcChiSquare(current) > getChiSquareChartValue(df)){
						for(int i = 0; i < current.children.length; i++){
							if(current.children[i] != null){
								tree.add(current.children[i]);
							}
							
							current.attributeIndex = bestAttributeIndex;
						}
					} else{
						// System.out.println("test");
						current.children = null;
					}
				}else{
					for(int i = 0; i < current.children.length; i++){
						if(current.children[i] != null){
							tree.add(current.children[i]);
						}
						
						current.attributeIndex = bestAttributeIndex;
					}
				}
			}
		}
	}
	
	 public void printTree() {        
		 if (rootNode != null) {         
			 rootNode.printNode("");	        
		 } 
	 }
	 
	/*
	 * returns relevant chi square value from the chi square chart
	 * according to df and p-value
	 */
	private double getChiSquareChartValue(int df) {
		
		// possible values for each p_value/alpa risk
		double[] column005 = {7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300, 29.819, 31.319};
		double[] column05 = {3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026, 22.362, 23.685};
		double[] column25 = {1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845, 15.984, 17.117};
		double[] column5 = {0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340, 12.340, 13.339};
		double[] column75 = {0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438, 9.299, 10.165};

		//System.out.println(m_p_value);
		if (m_p_value == 0.75) {
			return column75[df-1];
		}
		else if  (m_p_value == 0.5) {
			return column5[df-1];
		}
		else if  (m_p_value == 0.25) {
			return column25[df-1];
		}
		else if  (m_p_value == 0.05) {
			return column05[df-1];
		}
		else if  (m_p_value == 0.005) {
			return column005[df-1];
		}
		 return 0;
	}

	/*
	 * Updates positive instances, negative instances and numOfInstances for each node
	 */	
	private int updateInstancesForNode(Node node){

		for(int i =0; i < node.instances.numInstances(); i++){
			Instance instance = node.instances.get(i);
			if(instance != null){
				int instanceType = (int)(node.instances.get(i).classValue());
				if(instanceType == 0){
					node.typeOneInstances++;
				}else if(instanceType == 1){
					node.typeTwoInstances++;
				}			
			}		
		}
		
		node.numberOfInstances = node.typeOneInstances + node.typeTwoInstances;
		if(node.typeOneInstances < node.typeTwoInstances) node.returnValue = 1;
		if(node.numberOfInstances == 0){		
			return -1;	
		}
		return 0;
	}
	
	/*
	 * Finds the best attribute for the next distribution
	 * 
	 * @param node, dataSet
	 */
	private int getBestAttribute(Node node, Instances dataSet){
		
		double maxGain = 0;
		int bestAttribute = -1;
		
		for(int i = 0; i < (this.rootNode.instances.numAttributes()- 1); i++){
			
			createChildren(node, i, dataSet);
			
			double currentGain = calcGain(node);
			
			if(currentGain > maxGain){
				bestAttribute = i;
				maxGain = currentGain;
			}
		}
		
		node.children = null; // ???
		return bestAttribute;
	}
	
	/*
	 * Creates children for a node based on the number of values in the chosen attribute,
	 * then distributes the node's instances between the children
	 */
    private void createChildren(Node node, int attributeIndex, Instances dataSet) {
    	
		int numOfChildren = this.rootNode.instances.attribute(attributeIndex).numValues();
		node.children = new Node[numOfChildren];
		
		// create the children
		for(int i = 0; i < numOfChildren; i++){
			node.children[i] = new Node();
			node.children[i].parent = node;
			node.children[i].instances = new Instances(dataSet, dataSet.numInstances());
			
		}
		
		// distribute between the children the instances
		for(int i = 0; i < node.numberOfInstances; i++){
			
			Instance currentInstance = node.instances.get(i);
			
			if (currentInstance != null){
				
				int childIndex = (int) currentInstance.value(attributeIndex);
				node.children[childIndex].instances.add(currentInstance);
			}
		}
		
		for(int i = 0; i < numOfChildren; i++){
			if(node.children[i] != null){
				int check = updateInstancesForNode(node.children[i]);
				if(check == -1){
					// System.out.println("null");
					node.children[i] = null;
				}else{
					// System.out.println("test");
				}
			}
		}
	}

    @Override
   	public double classifyInstance(Instance instance) {
       	return classifyInstance(rootNode, instance);
       }
       
       
   	public double classifyInstance(Node node, Instance instance) {
       	double val = node.returnValue;
   	    int attributeToCheck = node.attributeIndex;
   	    int attributeValue = (int) instance.value(attributeToCheck);
   	    if (node.children != null) {
   	        if (node.children[attributeValue] != null){
   	            val = classifyInstance(node.children[attributeValue], instance);
   	        }
   	    }
   	    return val;
       }
       
       
       
   	private double calcGain(Node node) {
   		// Using Entropy
   		if (m_ImpurityMeasure == ImpurityMeasure.Entropy) {
   			double info_gain = calcEntropy(node);

   			for (int i = 0; i < node.children.length; i++) {
   				if (node.children[i] != null) {
   					double child_instances = (double) (node.children[i].numberOfInstances)
   							/ (node.numberOfInstances);
   					double entropy_of_child = calcEntropy(node.children[i]);
   					double mult = child_instances * entropy_of_child;
   					info_gain -= mult;
   				}
   			}
   			return info_gain;
   		}
   		// Using Gini
   		else if (m_ImpurityMeasure == ImpurityMeasure.Gini) {
   			double gini_gain = calcGini(node);
   			
   			for (int i = 0; i < node.children.length; i++) {
   				if (node.children[i] != null) {
   					double child_instances = (double) (node.children[i].numberOfInstances)
   							/ (node.numberOfInstances);
   					double gini_of_child = calcGini(node.children[i]);
   					double mult = child_instances * gini_of_child;
   					gini_gain -= mult;
   				}
   			}
   			return gini_gain;
   		}
   		
   		return 0;
   	}
   	  	
       public double calcAvgError(Instances dataSet) {
   		int amountOfInstances = dataSet.numInstances();
   		double calculatedClass, realClass;
   		double classificationMistakes = 0;
   		for (int i = 0; i < amountOfInstances; i++) {
   			realClass = dataSet.instance(i).classValue();
   			calculatedClass = classifyInstance(dataSet.instance(i));
   			if (realClass != calculatedClass)
   				classificationMistakes++;
   		}

   		return (classificationMistakes / amountOfInstances);
   	}
       
       
       
       private double calcChiSquare(Node node){
       	double chiSquare = 0;
       	
       	for(int i = 0; i < node.children.length; i++){
       		
       		if(node.children[i] != null){
       			
       				double e_1 = node.children[i].numberOfInstances
   							* ((double) node.typeOneInstances / node.numberOfInstances);
   					double p_1 = (double) node.children[i].typeTwoInstances;
   					chiSquare += (Math.pow(p_1 - e_1, 2)) / e_1;
   					
   					double e_2 = node.children[i].numberOfInstances
   							* ((double) node.typeOneInstances / node.numberOfInstances);
   					double p_2 = (double) node.children[i].typeTwoInstances;
   					chiSquare += (Math.pow(p_2 - e_2, 2)) / e_2;
   					
       		}
       	}
       	
       	return chiSquare;
       }
       
    private int getDf(Node node) {
   		int df = 0;
   		for (int i = 0; i < node.children.length; i++) {
   			if (node.children[i] != null) {
   				df++;
   			}
   		}
   		return df - 1;
    }
       
   	private double calcEntropy(Node node) {
   		double entropy = 0;

   		double prob_of_type1 = (node.typeOneInstances)/(node.numberOfInstances);
   		double prob_of_type2 = (node.typeTwoInstances)/(node.numberOfInstances);
   		   		
   		entropy += ((-1) * prob_of_type1 * this.log2(prob_of_type1)) 
   				+  ((-1) * prob_of_type2 * this.log2(prob_of_type2));
   			
	
   		return entropy;
   	}
   	
   	private double calcGini(Node node) {
   		
   		double giniSum = 0;

   		double prob_of_type1 = Math.pow((node.typeOneInstances)/(node.numberOfInstances),2);
   		double prob_of_type2 = Math.pow((node.typeTwoInstances)/(node.numberOfInstances),2);
   		giniSum += prob_of_type1 + prob_of_type2;


   		return (1-(giniSum));
   	}

   	
   	private double log2(double number) {
   		return Math.log(number) / Math.log(2);
   	}
    
    public void setImpurityMeasure(ImpurityMeasure im){
    	m_ImpurityMeasure = im;
    }
    
	public void setPValue(double p_value) {
		m_p_value = p_value;
	}
	
	public Node getRootNode() {
        return rootNode;
    }
	
    
    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}

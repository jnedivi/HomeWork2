package HomeWork2;

import weka.classifiers.Classifier;
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
}

public class DecisionTree implements Classifier {
	private Node rootNode;
	
	public enum ImpurityMeasure{
		Entropy, Gini, Null
	};
	
	public enum Pruning{
		No, Yes
	};
	
	private ImpurityMeasure m_ImpurityMeasure = ImpurityMeasure.Null;
	
	private Pruning m_Pruning = Pruning.No;
	
	public static final double[] k_p_values = {0.005, 0.05, 0.25, 0.5, 0.75, 1};
	
	private double m_BestPvalue;
	
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		this.buildTree(arg0);
	}
    
	public void buildTree(Instances dataSet) throws Exception {
		this.rootNode = new Node();
		Queue<Node> tree = new LinkedList<Node>();
		tree.add(this.rootNode);
		this.rootNode.instances = new Instances(dataSet, dataSet.numInstances());
		updateInstancesForNode(this.rootNode);
		
		while(!tree.isEmpty()){
			Node current = tree.poll();
			int bestAttributeIndex = getBestAttribute(current, dataSet);
			
			if(bestAttributeIndex != -1){
				createChildren(current, bestAttributeIndex, dataSet);
				
				int df = getDf(current);
				if(calcChiSquare(current) > getChiSquareValue(df)){
					for(int i = 0; i < current.children.length; i++){
						if(current.children[i] != null){
							tree.add(current.children[i]);
							current.attributeIndex = bestAttributeIndex;
						}
					}
				} else current.children = null;
			}		
		}
	}
	/*
	 * returns relevant chi square value from the chi square chart
	 * according to df and p-value
	 */
	private double getChiSquareValue(int df) {
		
		// possible values for each p_value/alpa risk
		double[] op0_005 = {7.879, 10.597, 12.838, 14.860, 16.750, 18.548, 20.278, 21.955, 23.589, 25.188, 26.757, 28.300};
		double[] op0_05 = {3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026};
		double[] op0_25 = {1.323, 2.773, 4.108, 5.385, 6.626, 7.841, 9.037, 10.219, 11.389, 12.549, 13.701, 14.845};
		double[] op0_5 = {0.455, 1.386, 2.366, 3.357, 4.351, 5.348, 6.346, 7.344, 8.343, 9.342, 10.341, 11.340};
		double[] op0_75 = {0.102, 0.575, 1.213, 1.923, 2.675, 3.455, 4.255, 5.071, 5.899, 6.737, 7.584, 8.438};
		
		if (m_BestPvalue == 0.75) {
			return op0_75[df-1];
		}
		else if  (m_BestPvalue == 0.5) {
			return op0_5[df-1];
		}
		else if  (m_BestPvalue == 0.25) {
			return op0_25[df-1];
		}
		else if  (m_BestPvalue == 0.05) {
			return op0_05[df-1];
		}
		else if  (m_BestPvalue == 0.005) {
			return op0_005[df-1];
		}
		return 0;
	}

	/*
	 * Updates positive instances, negative instances and numOfInstances for each node
	 */	
	private int updateInstancesForNode(Node node){
		int i = 0;
		while(i < node.instances.numInstances()){
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
		if(node.numberOfInstances == 0) return -1;
		return 0;
	}
	/*
	 * Finds the best attribute for the next distribution
	 * 
	 * @param node
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
	 * 
	 */
    private void createChildren(Node node, int attributeIndex, Instances dataSet) {
    	
		int numOfChildren = this.rootNode.instances.attribute(attributeIndex).numValues();
		node.children = new Node[numOfChildren];
		
		// create the children
		for(int i = 0; i < node.children.length; i++){
			node.children[i] = new Node();
			node.children[i] = node;
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
		
		for(int i = 0; i < node.children.length; i++){
			if(node.children[i] != null){
				if(updateInstancesForNode(node) != -1){
					node.children[i] = null;
				}
			}
		}
	}

	@Override
	public double classifyInstance(Instance instance) {

    }
    
    private double calcChiSquare(Node node){
    	double chiSquare = 0;
    	
    	for(int i = 0; i < node.children.length; i++){
    		
    		if(node.children[i] != null){
    			for(int j = 0; j < 2; j++){
    				
    			}
    		}
    	}
    	return chiSquare;
    }
    
    private int getDf(Node node){
    	return node.numberOfInstances -1;
    }
    
    public void setImpurityMeasure(ImpurityMeasure im){
    	m_ImpurityMeasure = im;
    }
    
    public void setPruning(Pruning status){
    	m_Pruning = status;
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

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
	
	private ImpurityMeasure m_ImpurityMeasure = ImpurityMeasure.Null;
	
	public double[] p_values = {0.005, 0.05, 0.25, 0.5, 0.75, 1};
	
	private double m_Best_p_value;
	
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		this.buildTree(arg0);
	}
    
	public void buildTree(Instances dataSet) throws Excpetion {
		this.rootNode = new Node();
		Queue<Node> tree = new LinkedList<Node>();
		tree.add(this.rootNode);
		this.rootNode.instances = new Instances(dataSet, dataSet.numInstances());
		updateInstancesForNode(this.rootNode);
		
	}
	
	/*
	 * Updates positive instances, negative instances and numOfInstances for each node
	 */	
	private void updateInstancesForNode(Node node){
		int i = 0;
		while(i < node.instances.numInstances()){
			Instance instance = node.instances.get(i);
			if(instance != null){
				int instanceType = (int) (node.instances.get(i).classValue());
				
				if(instanceType == 0){
					node.typeOneInstances++;
				}else if(instanceType == 1){
					node.typeTwoInstances++;
				}			
			}			
		}
		
		node.numberOfInstances = node.typeOneInstances + node.typeTwoInstances;
		if(node.typeOneInstances < node.typeTwoInstances) node.returnValue = 1;
		
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

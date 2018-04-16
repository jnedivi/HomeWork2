package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import HomeWork2.DecisionTree.ImpurityMeasure;
import weka.core.Instances;
import HomeWork2.DecisionTree;


public class MainHW2 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
	
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
        //TODO: complete the Main method
		DecisionTree treeEntropy = new DecisionTree();
		treeEntropy.setImpurityMeasure(ImpurityMeasure.Entropy);
		treeEntropy.buildClassifier(trainingCancer);
		double treeEntropyError = treeEntropy.calcAvgError(validationCancer);
		System.out.println("Validation error using Entropy: " + treeEntropyError);
		
		DecisionTree treeGini = new DecisionTree();
		treeGini.setImpurityMeasure(ImpurityMeasure.Gini);
		treeGini.buildClassifier(trainingCancer);
		double treeGiniError = treeGini.calcAvgError(validationCancer);
		System.out.println("Validation error using Gini: " + treeGiniError);
		
		ImpurityMeasure bestMeasure = ImpurityMeasure.Entropy;
		if(treeGiniError < treeEntropyError) {
			bestMeasure = ImpurityMeasure.Gini;
		}
		
		double best_validation_error = 1;
		double best_p_value = 0;
		DecisionTree best_tree = new DecisionTree();
		// Iterating over every p_value and prune accordingly
		for (int i = 0; i < treeEntropy.k_p_values.length; i++) {
			
			DecisionTree tree = new DecisionTree();
			tree.setImpurityMeasure(bestMeasure);
			tree.setPValue(treeEntropy.k_p_values[i]);
			
			tree.buildClassifier(trainingCancer);
			System.out.println("");
			System.out.println("Decision Tree with p_value of: " + treeEntropy.k_p_values[i]);
			double trainError = tree.calcAvgError(trainingCancer);
			System.out.println("The training error of the decision tree is: " + trainError);
			
			int maxHeight = findMaxTreeHeight(tree);
            System.out.println("Max height on validation data: " + maxHeight);
            int avgHeight = findAvgTreeHeight(tree);
            System.out.println("Average height on validation data: " + avgHeight);
			
			double validationError = tree.calcAvgError(validationCancer);
			System.out.println("The validation error of the decision tree is: " + validationError);
			
			if (validationError < best_validation_error) {
				best_validation_error = validationError;
				best_p_value = treeEntropy.k_p_values[i];
				best_tree = tree;
			}
		}
		
		System.out.println("");
		System.out.println("Best validation error at p_value = : " + best_p_value);
		double testError = best_tree.calcAvgError(testingCancer);
		System.out.println("Test error with best tree: " + testError);
		System.out.println("Representation of the best tree by 'if statements'");
		best_tree.printTree();
	}
	
	private static int findAvgTreeHeight(DecisionTree tree) {
        // we use an array where the first cell is the number of leaves and the second cell is the sum of their heights
        int[] avg = new int[2];
        avg = findAvgLeafHeight(tree.getRootNode(), 0, avg);
        int numberOfleaves = avg[0];
        int sumOfLeavesHeights = avg[1];
        return sumOfLeavesHeights / numberOfleaves;
    }

    private static int[] findAvgLeafHeight(Node node, int leafHeight, int[] avg) {
        if (node.children == null) {
            // this is a leaf
            avg[0] += 1;
            avg[1] += leafHeight;
            return avg;
        }
        leafHeight++;
        for (int i = 0; i < node.children.length; i++) {
            Node child = node.children[i];
            if (child != null) {
                findAvgLeafHeight(child, leafHeight, avg);
            }
        }
        return avg;
    }

    private static int findMaxTreeHeight(DecisionTree tree) {
        return findMaxNodeHeight(tree.getRootNode(), 0);
    }

    private static int findMaxNodeHeight(Node node, int maxNodeHeight) {
    	//System.out.println(node.children.length);
        if (node.children == null) {
            return maxNodeHeight;
        }
        maxNodeHeight++;
        for (int i = 0; i < node.children.length; i++) {
            Node child = node.children[i];
            if (child != null) {
                maxNodeHeight = Math.max(maxNodeHeight, findMaxNodeHeight(child, maxNodeHeight));
            }
        }
        return maxNodeHeight;
    }
	
	
}

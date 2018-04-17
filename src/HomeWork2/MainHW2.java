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
		
        // Constructing a tree with Entropy as the impurity measure. 
		DecisionTree entropy = new DecisionTree();
		entropy.setImpurityMeasure(ImpurityMeasure.Entropy);
		entropy.buildClassifier(trainingCancer);
		// Calculate the average error on the validation set.
		double entropyError = entropy.calcAvgError(validationCancer);
		System.out.println("Validation error using Entropy is: " + entropyError);
		
		// Constructing a tree with Gini as the impurity measure. 
		DecisionTree gini = new DecisionTree();
		gini.setImpurityMeasure(ImpurityMeasure.Gini);
		gini.buildClassifier(trainingCancer);
		// Calculate the average error on the validation set.
		double giniError = gini.calcAvgError(validationCancer);
		System.out.println("Validation error using Gini is: " + giniError);
		
		// Choose the impurity measure that gave you the lowest validation error. 
		// Use this impurity measure for the rest of the tasks.
		ImpurityMeasure betterImpurityMeasure = ImpurityMeasure.Entropy;
		if(entropyError > giniError) betterImpurityMeasure = ImpurityMeasure.Gini;
		
		System.out.println("-----------------------------------------");
		
		
		
		

	}

	
	
}

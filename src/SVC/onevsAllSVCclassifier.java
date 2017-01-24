package SVC;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import Utils.SUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

public class onevsAllSVCclassifier extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;
	private String m_S = "overparamLR"; 					// -S (overparamLR, vanillaLR)

	private boolean m_Discretization	 	= false; 			// -D
	private boolean m_MVerb 					= false; 			// -V		
	private boolean m_Regularization      = false;            // -R
	private double m_Lambda = 0.001;                           // -L

	private boolean m_ClassSpecification = false;          // -C

	private String m_O = "QN";                                       // -O (QN, CG, GD, Tron, SGD)

	private double[] probs;	

	private SVC[] svc;

	private int N;
	private int nc;

	private weka.filters.supervised.attribute.Discretize m_Disc = null;

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		Instances  m_DiscreteInstances = null;

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(instances);
			System.out.println("Applying Discretization Filter - dodo onevsAllSVC");
			m_DiscreteInstances = weka.filters.Filter.useFilter(instances, m_Disc);
			System.out.println("Done");

			m_Instances = new Instances(m_DiscreteInstances);
			m_DiscreteInstances = new Instances(m_DiscreteInstances, 0);
		} else {
			m_Instances = new Instances(instances);
			instances = new Instances(instances, 0);
		}

		N = m_Instances.numInstances();
		nc = m_Instances.numClasses();

		// Remove instances with missing class
		m_Instances.deleteWithMissingClass();

		if (m_S.equalsIgnoreCase("vanillaSVC")) {

			if (nc == 2) {

				svc = new SVC[1];

				svc[0] = new vanillaSVC(m_Instances, m_Regularization, m_Lambda, m_O);
				svc[0].train();

			} else if (nc > 2) {

				svc = new SVC[nc];
				Instances[] tempInstancesX = new Instances[nc];

				Instances genericHeader = getHeader(m_Instances);

				for (int c = 0; c < nc; c++) {
					/* Create a local copy of modified instances */
					tempInstancesX[c] = new Instances(genericHeader, 0);
					updateHeader(tempInstancesX[c], c, m_Instances);

					/* Create C different classifiers */				
					svc[c] = new vanillaSVC(tempInstancesX[c], m_Regularization, m_Lambda, m_O);
					svc[c].train();
				}
			}

		} else {
			System.out.println("m_S value should be from set: {vanillaSVC}");
		}

		// free up some space
		m_Instances = new Instances(m_Instances, 0);
	}


	@Override
	public double[] distributionForInstance(Instance instance) {

		if (m_Discretization) {
			synchronized(m_Disc) {	
				m_Disc.input(instance);
				instance = m_Disc.output();
			}
		}

		probs = new double[nc];

		if (nc == 2) {
			
			double score = svc[0].predict(instance);
			if (score >= 0) {
				probs[0] = 1;
			} else {
				probs[1] = 1;
			}

		} else if (nc > 2) {

			double[] score = new double[nc];

			for (int c = 0; c < nc; c++) {
				score[c] = svc[c].predict(instance);			
			}

			int winner = SUtils.maxLocationInAnArray(score);
			probs[winner] = 1;
		}

		return probs;
	}	

	public static Instances getHeader(Instances instances) {
		Instances header = null;
		int n = instances.numAttributes() - 1;
		ArrayList<Attribute> attlist = new ArrayList<Attribute>(n + 1);

		for (int i = 0; i < n; i++) {
			attlist.add(instances.attribute(i));
		}

		String className = instances.classAttribute().name();
		List<String> classNamesList = new ArrayList<String>(2);
		for (int i = 0; i < 2; i++) {
			classNamesList.add(i+"");
		}
		Attribute classAtt = new Attribute(className, classNamesList);

		attlist.add(classAtt);

		header = new Instances(instances.relationName(), attlist, 0);
		header.setClassIndex(n);

		return header;
	}

	public static void updateHeader(Instances header, int classVal, Instances instances) {

		int n = instances.numAttributes();

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);
			int x_C = (int) inst.classValue();

			double[] instanceValues = new double[n];
			for (int ii = 0; ii < n - 1; ii++) {
				instanceValues[ii] = inst.value(ii);
			}

			if (x_C == classVal) {
				instanceValues[n - 1] = 0;	
			} else {
				instanceValues[n - 1] = 1;
			}

			DenseInstance denseInstance = new DenseInstance(1.0, instanceValues);
			header.add(denseInstance);
		}

	}


	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {

		m_Discretization = Utils.getFlag('D', options);
		m_MVerb = Utils.getFlag('V', options);

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		String SO = Utils.getOption('O', options);
		if (SO.length() != 0) {
			m_O = SO;
		}

		m_Regularization = Utils.getFlag('R', options);
		if (m_Regularization) {
			String SL = Utils.getOption('L', options);
			if (SL.length() != 0) {
				m_Lambda = Double.parseDouble(SL);
			}	
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new onevsAllSVCclassifier(), argv);
	}

}

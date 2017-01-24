package ANN;

import java.util.Arrays;

import Utils.SUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.Bagging;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

public class ANNclassifier extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	private boolean m_Discretization	 	= false; 			// -D
	private boolean m_Regularization      = false;            // -R
	private double m_Lambda = 0.001;                           // -L

	private String m_O = "QN";                                       // -O (QN, CG, GD, Tron, SGD)

	private double[] probs;	

	private ANN ann;

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
			System.out.println("Applying Discretization Filter - dodo ANN");
			m_DiscreteInstances = weka.filters.Filter.useFilter(instances, m_Disc);
			System.out.println("Done");

			m_Instances = new Instances(m_DiscreteInstances);
			m_DiscreteInstances = new Instances(m_DiscreteInstances, 0);
		} else {
			m_Instances = new Instances(instances);
			instances = new Instances(m_Instances);
		}

		System.out.println("N = " + m_Instances.numInstances() + ", n  = " + (m_Instances.numAttributes()-1) + " c = ," + m_Instances.numClasses());

		// Remove instances with missing class
		m_Instances.deleteWithMissingClass();

		ann = new overparamANN(m_Instances, m_Regularization, m_Lambda, m_O);
		ann.train();

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

		probs = ann.predict(instance);
		SUtils.exp(probs);

		return probs;
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
		runClassifier(new ANNclassifier(), argv);
	}

}

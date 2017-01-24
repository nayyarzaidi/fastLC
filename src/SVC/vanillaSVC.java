package SVC;

import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;

public class vanillaSVC extends SVC {

	public vanillaSVC(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "v");

		for (int u = 0; u < n; u++) {
			if (instances.attribute(u).isNominal()) {
				isNumericTrue[u] = false;
				paramsPerAtt[u] = instances.attribute(u).numValues();
			} else if (instances.attribute(u).isNumeric()) {
				isNumericTrue[u] = true;
				paramsPerAtt[u] = 1;
			}
		}

		startPerAtt = new int[n];

		np = 1;
		for (int u = 0; u < n; u++) {
			startPerAtt[u] += np;
			if (instances.attribute(u).isNominal()) {
				np += paramsPerAtt[u];
			} else if (instances.attribute(u).isNumeric()) {
				np += 1;
			}
		}

		parameters = new double[np];
		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);

		if (m_O.equalsIgnoreCase("Tron")) {
			Dbin = new double[N];	
		}

	}

	public void train() {
		super.train();
		instances = new Instances(instances, 0);
	}

	public double predict(Instance inst) {
		double score = 0;

		score = parameters[0];

		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				if (isNumericTrue[u]) {
					int pos = getNumericPosition(u);
					score += (parameters[pos] * uval);
				} else {
					int pos = getNominalPosition(u, (int) uval);
					score += parameters[pos];
				}
			}
		}

		return score;
	}

	public void computeGrad(Instance inst, double v, double[] gradients) {

		gradients[0] += v;
				
		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				for (int c = 0; c < nc - 1; c++) {
					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u);
						gradients[pos] += v * uval;
					} else {
						int pos = getNominalPosition(u, (int) uval);
						gradients[pos] += v;
					}
				}
			}
		}

	}

	public void computeHessian(int i, double[] probs) {
		/* 
		 * Only present due to interace implementation. 
		 * Not needed for SVC
		 */

	}

	@Override
	public void computeHv(double[] s, double[] Hs) {
		/* 
		 * Implementing this function in Parent class
		 */
	}


}

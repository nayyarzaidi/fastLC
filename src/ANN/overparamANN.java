package ANN;

import java.util.Arrays;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class overparamANN extends ANN {

	public overparamANN(Instances instances, boolean regularization, double lambda, String m_O) {

		super(instances, regularization, lambda,  m_O, "op");

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

		np = nc;
		for (int u = 0; u < n; u++) {
			startPerAtt[u] += np;
			if (instances.attribute(u).isNominal()) {
				np += (paramsPerAtt[u] * nc);
			} else if (instances.attribute(u).isNumeric()) {
				np += (1 * nc);
			}
		}

		parameters = new double[np];
		System.out.println("Model is of Size: " + np);
		Arrays.fill(parameters, 0.0);

		if (m_O.equalsIgnoreCase("Tron")) {
			D = new double[N][nc][nc];
		}

	}

	public void train() {
		super.train();
		instances = new Instances(instances, 0);
	}

	public double[] predict(Instance inst) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = parameters[c];

			for (int u = 0; u < n; u++) {
				if (!inst.isMissing(u)) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						probs[c] += (parameters[pos] * uval);
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						probs[c] += parameters[pos];
					}
				}
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public void computeGrad(Instance inst, double[] probs, int x_C, double[] gradients) {

		for (int k = 0; k < nc; k++) {
			for (int c = 0; c < nc; c++) {
				gradients[k] += ((SUtils.ind(c, x_C) - probs[c]) * ((-1) * (SUtils.ind(k, c) - probs[k]) * probs[c]));
			}
		}

		for (int u = 0; u < n; u++) {
			if (!inst.isMissing(u)) {
				double uval = inst.value(u);

				for (int k = 0; k < nc; k++) {

					for (int c = 0; c < nc; c++) {
						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u, k);
							gradients[pos] += ((SUtils.ind(c, x_C) - probs[c]) * ((-1) * (SUtils.ind(k, c) - probs[k]) * probs[c]) * uval);
						} else {
							int pos = getNominalPosition(u, (int) uval, k);
							gradients[pos] += ((SUtils.ind(c, x_C) - probs[c]) * ((-1) * (SUtils.ind(k, c) - probs[k]) * probs[c]));
						}
					}

				}
			}
		}

	}

	public void computeHessian(int i, double[] probs, int x_C) {

		for (int k1 = 0; k1 < nc; k1++) {
			for (int k2 = 0; k2 < nc; k2++) {

				for (int c = 0; c < nc; c++) {
					
					D[i][k1][k2] += ( 
							(-1) * 
							((-1) * (SUtils.ind(k2, c) - probs[k2]) * probs[c]) * ((SUtils.ind(k1, c) - probs[k1]) * probs[c]) + 
							((SUtils.ind(x_C, c) - probs[c])) * ((-1) * (SUtils.ind(k1, k2) - probs[k2]) * probs[k1]) * probs[c] +
							((SUtils.ind(x_C, c) - probs[c])) * ((SUtils.ind(k1, c) - probs[k1])) * ((SUtils.ind(k2, c) - probs[k2]) * probs[c])
									);
					 
				}

			}			
		}

	}

	@Override
	public void computeHv(double[] s, double[] Hs) {

		double[] wa = new double[N * nc];
		double[] wa2 = new double[N * nc];

		int[] offset = new int[nc];
		int index = 0;
		for (int c = 0; c < nc; c++) {
			offset[c] = index;
			index += N;
		}

		//Xv(s, wa);
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);

			for (int c = 0; c < nc; c++) {

				wa[i + offset[c]] += s[c];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						wa[i] += (s[pos] * uval);
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						wa[i + offset[c]] += s[pos];
					}

				}

			}
		}

		//D[i] * wa[i];
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int c1 = 0; c1 < nc; c1++) {					
				for (int c2 = 0; c2 < nc; c2++) {
					wa2[i + offset[c1]] += (D[i][c1][c2] * wa[i + offset[c2]]);
				}
			}
		}

		//XTv(wa, Hs);
		for (int i = 0; i < np; i++) {
			Hs[i] = 0;
		}

		for (int i = 0; i < instances.numInstances(); i++) {
			Instance inst = instances.instance(i);

			for (int c = 0; c < nc; c++) {

				Hs[c] += wa2[i + offset[c]];

				for (int u = 0; u < n; u++) {
					double uval = inst.value(u);

					if (isNumericTrue[u]) {
						int pos = getNumericPosition(u, c);
						Hs[pos] += (wa2[i + offset[c]] * uval);
					} else {
						int pos = getNominalPosition(u, (int) uval, c);
						Hs[pos] += wa2[i + offset[c]];
					}
				}

			}
		}	

		//s[i] + Hs[i];
		for (int i = 0; i < np; i++) {
			Hs[i] = s[i] + Hs[i];
		}

	}

}

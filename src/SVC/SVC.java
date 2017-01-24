package SVC;

import optimize.DifferentiableFunction;
import optimize.FunctionValues;

import optimize.LBFGSBException;
import optimize.Result;
import optimize.StopConditions;

import optimize.Minimizer;
import optimize.MinimizerGD;
import optimize.MinimizerLBFGS;
import optimize.MinimizerTron;
import optimize.MinimizerCG;

import weka.core.Instance;
import weka.core.Instances;

public abstract class SVC {

	protected Instances instances;

	protected double[] parameters;
	protected double[] gradients;

	protected double[][][] D;
	protected double[] Dbin;

	protected int N;
	protected int nc;
	protected int n;

	protected int np = 0;

	protected int[] paramsPerAtt;
	protected int[] startPerAtt;

	protected boolean[] isNumericTrue;

	protected boolean regularization = false;
	protected double lambda = 0.001;

	protected String m_O = "QN";
	protected boolean is_Verbose = true;

	ObjectiveFunction function_to_optimize;

	private int maxIterations = 10000;	

	private String name;

	protected double  eps = 0.0001;

	double[] z;
	double[] y;
	int[] I;
	
	int sizeI;

	public SVC(Instances instances, boolean regularization, double lambda, String m_O, String name) {

		this.name = name;
		this.instances = instances;

		this.N = instances.numInstances();
		this.n = instances.numAttributes() - 1;
		this.nc = instances.numClasses();

		if (nc > 2) {
			System.out.println("The no. of classes for SVC has to be 2");
			System.exit(-1);
		}

		this.m_O = m_O;

		this.regularization = regularization;
		this.lambda = lambda;

		isNumericTrue = new boolean[n];
		paramsPerAtt = new int[n];

		function_to_optimize = new ObjectiveFunction();

		z = new double[N];
		y = new double[N];
		I = new int[N];

		for (int i = 0; i < N; i++) {
			Instance inst = instances.instance(i);
			int x_C = (int) inst.classValue();
			y[i] = (x_C == 0) ? 1 : -1;
		}
	}

	class ObjectiveFunction implements DifferentiableFunction {

		@Override
		public FunctionValues getValues(double[] params) {

			double f = 0.0;

			for (int i = 0; i < np; i++) {
				parameters[i] = params[i];
			}
			gradients = new double[np];

			for (int i = 0; i < np; i++) 
				f += parameters[i] * parameters[i];
			
			f /= 2.0;
			
			for (int i = 0; i < N; i++) {
				Instance inst = instances.instance(i);

				z[i] = y[i] * predict(inst);

				double d = 1 - z[i];
				if (d > 0) 
					f += d * d;
			}

			sizeI = 0;
			for (int i = 0; i < N; i++) {
				if (z[i] < 1) {
					z[sizeI] = y[i] * (z[i] - 1);
					I[sizeI] = i;
					sizeI++;
				}
			}

			for (int i = 0; i < sizeI; i++) {
				Instance inst = instances.instance(I[i]);
				computeGrad(inst, z[i], gradients);
			}

			for (int i = 0; i < np; i++) 
				gradients[i] = parameters[i] + 2 * gradients[i];
			
			
			return new FunctionValues(f, gradients);
		}

		@Override
		public double fun() {

			double f = 0.0;
			
			for (int i = 0; i < np; i++) 
				f += parameters[i] * parameters[i];
			
			f /= 2.0;
			
			for (int i = 0; i < N; i++) {
				Instance inst = instances.instance(i);
				z[i] = y[i] * predict(inst);

				double d = 1 - z[i];
				if (d > 0) 
					f += d * d;
			}

			return f;
		}

		@Override
		public double fun(double[] point) {

			double[] oldParameters = new double[np];
			System.arraycopy(parameters, 0, oldParameters, 0, np);

			for (int i = 0; i < np; i++) {
				parameters[i] = point[i];
			}

			double f = 0.0;
			
			for (int i = 0; i < np; i++) 
				f += parameters[i] * parameters[i];
			
			f /= 2.0;
			
			for (int i = 0; i < N; i++) {
				Instance inst = instances.instance(i);
				z[i] = y[i] * predict(inst);

				double d = 1 - z[i];
				if (d > 0) 
					f += d * d;
			}

			System.arraycopy(oldParameters, 0, parameters, 0, np);

			return f;
		}

		@Override
		public void grad(double[] grad) {

			for (int i = 0; i < np; i++) {
				grad[i] = 0;
			}

			sizeI = 0;
			for (int i = 0; i < N; i++) {
				if (z[i] < 1) {
					z[sizeI] = y[i] * (z[i] - 1);
					I[sizeI] = i;
					sizeI++;
				}
			}

			for (int i = 0; i < sizeI; i++) {
				Instance inst = instances.instance(I[i]);
				computeGrad(inst, z[i], grad);
			}

			for (int i = 0; i < np; i++) 
				grad[i] = parameters[i] + 2 * grad[i];

		}
		
		@Override
		public void grad(int ii, double[] grad) {
			
		}

		@Override
		public void Hv(double[] s, double[] Hs) {
			double[] wa = new double[sizeI];
			
			for (int i = 0; i < sizeI; i++) {
				wa[i] = 0;
				Instance inst = instances.instance(I[i]);
				
				//subXv(s, wa);
				wa[i] = s[0];
				for (int u = 0; u < n; u++) {
					if (!inst.isMissing(u)) {
						double uval = inst.value(u);

						if (isNumericTrue[u]) {
							int pos = getNumericPosition(u);
							wa[i] += (s[pos] * uval);
						} else {
							int pos = getNominalPosition(u, (int) uval);
							wa[i] += s[pos];
						}
					}
				}
				
			}
			
			//subXTv(wa, Hs);
			for (int i = 0; i < np; i++) {
				Hs[i] = 0;
			}
			
			for (int i = 0; i < sizeI; i++) {
				Instance inst = instances.instance(I[i]);
				
				Hs[0] += wa[i];
				
				for (int u = 0; u < n; u++) {
					if (!inst.isMissing(u)) {
						double uval = inst.value(u);

						for (int c = 0; c < nc - 1; c++) {
							if (isNumericTrue[u]) {
								int pos = getNumericPosition(u);
								Hs[pos] += wa[i] * uval;
							} else {
								int pos = getNominalPosition(u, (int) uval);
								Hs[pos] += wa[i];
							}
						}
					}
				}
				
			}
			
			for (int i = 0; i < np; i++)
				Hs[i] = s[i] + 2 * Hs[i];
			
		}

		@Override
		public int get_nr_variable() {
			return np;
		}
		
		@Override
		public int get_nr_instances() {
			return N;
		}

		@Override
		public void hessian(int ii, double[] hessian) {
			// TODO Auto-generated method stub
			
		}

	};

	public abstract double predict(Instance inst);
	public abstract void computeGrad(Instance inst, double v, double[] gradients);
	public abstract void computeHessian(int i, double[] probs);
	public abstract void computeHv(double[] s, double[] Hs);

	public double regularizeFunction() {
		double f = 0.0;
		for (int i = 0; i < np; i++) {
			f += lambda/2 * parameters[i] * parameters[i];
		}
		return f;
	}

	public void regularizeGradient(double[] grad) {
		for (int i = 0; i < np; i++) {
			grad[i] += lambda * parameters[i];
		}
	}

	public void train() {

		if (m_O.equalsIgnoreCase("QN")) {

			double maxGradientNorm = 1e-32;

			Minimizer alg = new Minimizer();
			StopConditions sc = alg.getStopConditions();
			sc.setMaxGradientNorm(maxGradientNorm);
			sc.setMaxIterations(maxIterations);

			Result result;
			try {		
				if (is_Verbose) {
					System.out.println();
					System.out.print("fx_QN_" + name + " = [");
					alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});
					result = alg.run(function_to_optimize, parameters);
					System.out.println("];");
					//System.out.println(result);
					System.out.println("NoIter = " + result.iterationsInfo.iterations); System.out.println();
				} else {
					result = alg.run(function_to_optimize, parameters);
					System.out.println("NoIter = " + result.iterationsInfo.iterations);
					//System.out.println(result);
				}
			} catch (LBFGSBException e) {
				e.printStackTrace();
			}

		} else if (m_O.equalsIgnoreCase("CG")) {

			MinimizerCG alg = new optimize.MinimizerCG();
			alg.setMaxIterations(maxIterations);

			Result result;

			if (is_Verbose) {
				System.out.print("fx_CG_" + name + " = [");
				result = alg.run(function_to_optimize, parameters);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		} else if (m_O.equalsIgnoreCase("GD")) {

			MinimizerGD alg = new MinimizerGD();
			alg.setMaxIterations(maxIterations);
			Result result;	

			if (is_Verbose) {
				System.out.print("fx_GD_" + name + " = [");
				result = alg.run(function_to_optimize, parameters);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		} else if (m_O.equalsIgnoreCase("Tron")) {

			MinimizerTron alg = new MinimizerTron();
			alg.setMaxIterations(maxIterations);
			Result result;	

			if (is_Verbose) {
				System.out.print("fx_Tron_" + name + " = [");
				result = alg.run(function_to_optimize, parameters, eps);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters, eps);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		}  else if (m_O.equalsIgnoreCase("LBFGS")) {

			MinimizerLBFGS alg = new MinimizerLBFGS();
			alg.setMaxIterations(maxIterations);
			Result result;	

			if (is_Verbose) {
				System.out.print("fx_LBFGS_" + name + " = [");
				result = alg.run(function_to_optimize, parameters, eps);
				System.out.println("];");
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
				System.out.println();
			} else {
				result = alg.run(function_to_optimize, parameters, eps);
				System.out.println("NoIter = " + result.iterationsInfo.iterations);
			}

		} else {
			System.out.println("Only QN, CG, GD, Tron, LBFGS are implemented");
			System.exit(-1);
		} 

	}

	public void setModel(double[] newArray) {
		for (int i = 0; i < np; i++) {
			parameters[i] = newArray[i];
		}
	}

	public double[] getModel() {
		return parameters;
	}

	public int getNP() {
		return np;
	}

	public int getNumericPosition(int u, int c) {
		return startPerAtt[u] + (paramsPerAtt[u] * c);
	}
	public int getNominalPosition(int u, int uval, int c) {
		return startPerAtt[u] + ((paramsPerAtt[u] * c) + uval);
	}
	public int getNominalPosition(int u, int uval, int c, int[] localStartPerAtt) {
		return localStartPerAtt[u] + ((paramsPerAtt[u] * c) + uval);
	}

	public int getNumericPosition(int u) {
		return startPerAtt[u];
	}
	public int getNominalPosition(int u, int uval) {
		return startPerAtt[u] + uval;
	}

	public double getEps() {
		return eps;
	}

	public void setEps(double eps) {
		this.eps = eps;
	}

	public int getMaxIterations() {
		return maxIterations;
	}
	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

}

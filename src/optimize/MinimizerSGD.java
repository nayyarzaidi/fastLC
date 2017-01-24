package optimize;

import java.io.PrintStream;
import java.util.Arrays;

public class MinimizerSGD {

	// Max iterations
	private int maxIterations = 1000;

	private FunctionValues fv = null;
	private int totalFunctionEvaluations = 0;	 

	private boolean verbose = true;

	public Result run(DifferentiableFunction fun_obj, double[] w, double method) {

		int n = fun_obj.get_nr_variable();
		int N = fun_obj.get_nr_instances();
		double[] g = null;
		double f = 0.0;
		int iter = 0;

		for (int i = 0; i < n; i++)
			w[i] = 0;

		if (method == 0) {
			// ***************************************************
			// Standard SGD, step size is fixed
			// ***************************************************

			double eta = 1e-2;

			iter = 0;
			f = fun_obj.fun(w);

			do {

				for (int ii = 0; ii < N; ii++) {
					g = new double[n];
					fun_obj.grad(ii, g);

					double stepSize = eta;
					for (int i = 0; i < n; i++) {
						w[i] = w[i] - stepSize * g[i];
					}
				}

				double fnew = fun_obj.fun(w);
				double actred = f - fnew;

				if (verbose) {
					//info("iter %2d act %5.3e f %5.3e |g| %5.3e %n", iter, actred, f, gnorm);
					System.out.println(iter + ", " + "Reduction = " + actred + ", f = " + f + ", fnew = " + fnew + " + gnorm = " + euclideanNorm(g));
				}

				iter++;
				f = fnew;

			} while (iter < maxIterations || euclideanNorm(g) < 1e-3);


		} else if (method == 1) {
			// ***************************************************
			// Decaying learning rates
			// ***************************************************

			double eta0 = 1e-2;
			double eta = 1e-3;
			double actred = 0;

			iter = 0;
			f = fun_obj.fun(w);

			do {

				for (int ii = 0; ii < N; ii++) {
					g = new double[n];
					fun_obj.grad(ii, g);

					double stepSize = eta0 /(1 + eta * ii);
					for (int i = 0; i < n; i++) {
						w[i] = w[i] - stepSize * g[i];
					}
				}

				double fnew = fun_obj.fun(w);
				actred = f - fnew;

				if (verbose) {
					System.out.println("Epoch = " + iter + ", " + "Reduction = " + actred + ", f = " + f + ", fnew = " + fnew + ", gnorm = " + euclideanNorm(g));
				}

				iter++;
				f = fnew;

			} while (iter < maxIterations || actred < 0.1 || euclideanNorm(g) < 1e-3);

		} else if (method == 2) {
			// ***************************************************
			// Adagrad (element-wise adaptive learning rates)
			// ***************************************************

			double eta0 = 0.01;
			double smoothingParameter = 1e-9;
			double actred = 0;

			double[] G = new double[n];

			iter = 0;
			f = fun_obj.fun(w);
			do {

				for (int ii = 0; ii < N; ii++) {
					g = new double[n];
					fun_obj.grad(ii, g);

					for (int i = 0; i < n; i++) {
						G[i] += ((g[i] * g[i]));
					}

					double stepSize[] = new double[n];
					for (int i = 0; i < n; i++) {
						stepSize[i] = eta0 / (smoothingParameter + Math.sqrt(G[i]));

						if (stepSize[i] == Double.POSITIVE_INFINITY) {
							stepSize[i] = 0.0;
						}
					}

					for (int i = 0; i < n; i++) 
						w[i] = w[i] - stepSize[i] * g[i];

				}

				double fnew = fun_obj.fun(w);
				actred = f - fnew;

				if (verbose) {
					System.out.println("Epoch = " + iter + ", " + "Reduction = " + actred + ", f = " + f + ", fnew = " + fnew + ",  gnorm = " + euclideanNorm(g));
				}

				iter++;
				f = fnew;

			} while (iter < maxIterations && Math.abs(actred) > 0.01 && euclideanNorm(g) > 1e-4);


		} else if (method == 3) {
			// ***************************************************
			// Adadelta
			// ***************************************************

			double rho = 0.9;
			double smoothingParameter = 1e-9;

			double[] G = new double[n];
			double[] A = new double[n];
			double[] l = new double[n];

			double actred = 0;

			iter = 0;
			f = fun_obj.fun(w);
			do {

				for (int ii = 0; ii < N; ii++) {
					g = new double[n];
					fun_obj.grad(ii, g);

					for (int i = 0; i < n; i++) {
						G[i] = rho * G[i] + (1.0 - rho) * g[i] * g[i];

						l[i] = (Math.sqrt(A[i] + smoothingParameter)) / ( Math.sqrt(G[i] + smoothingParameter));
						double update = l[i] * g[i];
						A[i] = rho * A[i] + (1.0 - rho) * update * update;
					}

					double stepSize[] = new double[n];
					for (int i = 0; i < n; i++) {
						stepSize[i] =  (Math.sqrt(A[i] + smoothingParameter)) / ( Math.sqrt(G[i] + smoothingParameter));

						assert !Double.isNaN(stepSize[i]);
						assert !Double.isInfinite(stepSize[i]);
					}

					for (int i = 0; i < n; i++) 
						w[i] = w[i] - stepSize[i] * g[i];

				}

				double fnew = fun_obj.fun(w);
				actred = f - fnew;

				if (verbose) {
					System.out.println("Epoch = " + iter + ", " + "Reduction = " + actred + ", f = " + f + ", fnew = " + fnew + ",  gnorm = " + euclideanNorm(g));
				}

				iter++;
				f = fnew;

			} while (iter < maxIterations && Math.abs(actred) > 0.01 && euclideanNorm(g) > 1e-4);


		} else if (method == 4) {
			// ***************************************************
			// Amari's natural gradient descent
			// ***************************************************


		} else if (method == 5) {
			// ***************************************************
			// Pesky learning rate
			// ***************************************************

			double[] h = null;

			double slow_constant = 2;
			int init_samples = 10;

			double epsilon = 1e-9;
			int outlier_level = 1;

			double actred = 0;

			double[] gbar = new double[n];
			double[] vbar = new double[n];
			double[] hbar = new double[n];
			double[] vpart = new double[n];
			double[] taus = new double[n];

			if (slow_constant == 0)
				slow_constant = Math.max(1, (int)n/10);

			if (init_samples > 0) {

				g = new double[n];
				h = new double[n];

				for (int ii = 0; ii < init_samples; ii++) {
					for (int jj = ii; jj < ii + init_samples; jj++) {
						// Accumulate gradients
						fun_obj.grad(jj, g);
						fun_obj.hessian(jj, h);
					}
				}
				
				

			} else {
				for (int i = 0; i < n; i++) {
					gbar[i] = 0;
					vbar[i] = 1.0 * epsilon;
					hbar[i] = 1.0;
				}
			}

			for (int i = 0; i < n; i++) {
				vpart[i] = (gbar[i] * gbar[i]) / vbar[i];
				taus[i] = (1.0 + epsilon) * 2;
			}

			iter = 0;
			f = fun_obj.fun(w);
			do {

				for (int ii = init_samples; ii < N; ii += init_samples) {

					g = new double[n];
					h = new double[n];

					for (int jj = ii; jj < ii + init_samples; jj++) {
						// Accumulate gradients
						fun_obj.grad(jj, g);
						fun_obj.hessian(jj, h);
					}

					for (int i = 0; i < n; i++) {
						gbar[i] = (1 - 1/taus[i]) * gbar[i] + 1/taus[i] * g[i];
						vbar[i] = (1 - 1/taus[i]) * vbar[i] + 1/taus[i] * g[i] * g[i];
						hbar[i] = (1 - 1/taus[i]) * hbar[i] + 1/taus[i] * h[i];

						vpart[i] = 0;
						vpart[i] = vpart[i] + (gbar[i] * gbar[i]) / vbar[i];

						taus[i] = (1 - vpart[i]) * taus[i];
						taus[i] += (1 + epsilon);
					}

					double stepSize[] = new double[n];
					for (int i = 0; i < n; i++) {
						stepSize[i] =  vpart[i] /(hbar[i] + epsilon);
					}

					for (int i = 0; i < n; i++) 
						w[i] = w[i] - stepSize[i] * g[i];

				}

				double fnew = fun_obj.fun(w);
				actred = f - fnew;

				if (verbose) {
					System.out.println("Epoch = " + iter + ", " + "Reduction = " + actred + ", f = " + f + ", fnew = " + fnew + ", gnorm = " + euclideanNorm(g));
				}

				iter++;
				f = fnew;

			} while (iter < maxIterations); //while (iter < maxIterations && Math.abs(actred) > 0.01 && euclideanNorm(g) > 1e-4);


		}

		if (verbose)
			System.out.println("All Done");

		IterationsInfo info = null;
		info = new IterationsInfo(iter-1, totalFunctionEvaluations, IterationsInfo.StopType.MAX_ITERATIONS, null);	

		Result result = new Result(w, f, g, info);
		return result;
	}

	public void setMaxIterations(int m_MaxIterations) {
		maxIterations = 	m_MaxIterations;
	}

	private double euclideanNorm(double vector[]) {

		int n = vector.length;
		//		double mag = 0;
		//		
		//		for (int i = 0; i < n; i++) {
		//			mag += vector[i] * vector[i];
		//		}
		//		
		//		return Math.sqrt(mag);

		if (n < 1) {
			return 0;
		}

		if (n == 1) {
			return Math.abs(vector[0]);
		}

		double scale = 0; 
		double sum = 1; 

		for (int i = 0; i < n; i++) {
			if (vector[i] != 0) {
				double abs = Math.abs(vector[i]);

				if (scale < abs) {
					double t = scale / abs;
					sum = 1 + sum * (t * t);
					scale = abs;
				} else {
					double t = abs / scale;
					sum += t * t;
				}
			}
		}

		return scale * Math.sqrt(sum);
	}

	public void info(String format, Object... args) {
		PrintStream DEBUG_OUTPUT = System.out;
		if (DEBUG_OUTPUT == null) 
			return;
		DEBUG_OUTPUT.printf(format, args);
		DEBUG_OUTPUT.flush();

	}

}



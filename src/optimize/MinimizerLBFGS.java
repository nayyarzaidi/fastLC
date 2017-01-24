package optimize;

import java.io.PrintStream;

public class MinimizerLBFGS {

	// Max iterations
	private int maxIterations = 1000;

	private FunctionValues fv = null;
	private int totalFunctionEvaluations = 0;	 

	private boolean verbose = false;

	public Result run(DifferentiableFunction fun_obj, double[] w, double eps) {
		
		double f = 0.0;
		
		int ndim = fun_obj.get_nr_variable();
		double[] diag = new double [ndim];
		double[] g = new double[ndim];
		
		double[] x = new double[ndim];
		
		int iflag[] = new int[1];
		int iprint[] = new int [2];
		iprint [0] = 1;
		iprint [1] = 0;
		int m = 5;
		boolean diagco;
		diagco= false;
		
		eps = 1.0e-5;
		double xtol = 1.0e-16;
		iflag[0] = 0;
		
		int iter = 0;
		do {
			try {
				FunctionValues fv = fun_obj.getValues(x);
				f = fv.functionValue;
				g = fv.gradient;
				
				//System.out.print(f + ", ");
				
				LBFGS.lbfgs (ndim , m , x , f , g , diagco , diag , iprint , eps , xtol , iflag);
			} catch (LBFGS.ExceptionWithIflag e) {
				System.err.println( "MinimizerLBFGS: lbfgs failed.\n"+e );
			}

			iter += 1;
		}
		while ( iflag[0] != 0 && iter <= maxIterations );
		
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

}



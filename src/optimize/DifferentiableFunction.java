package optimize;

/**
 * Class responsible for returning the function values at given point
 * @author Mateusz Kobos
 */
public interface DifferentiableFunction {
	/**
	 * @param point point of the function evaluation
	 * @return values in given point if the algorithm should continue 
	 * computations or null if the algorithm should stop
	 */
	FunctionValues getValues(double[] point);
	
	double fun();
	double fun(double[] point);
	
	void grad( double[] grad);
	void grad(int inst, double[] g);
	void hessian(int ii, double[] hessian);
	
	void Hv(double[] d, double[] Hd);
	
	int get_nr_variable();

	int get_nr_instances();
}

package optimize;

import java.util.ArrayList;

public class SampleRun {
	public static void main(String[] args){
		try {
			QuadraticFun fun = new QuadraticFun();
			Minimizer alg = new Minimizer();
//			alg.setIterationFinishedListener(new SampleListener());
			ArrayList<Bound> bounds = new ArrayList<Bound>();
			bounds.add(new Bound(new Double(10), null));
			alg.setBounds(bounds);
			Result result = alg.run(fun, new double[]{40});
			System.out.println("The final result: "+result);
		} catch (LBFGSBException e) {
			e.printStackTrace();
		}
	}
}

class QuadraticFun implements DifferentiableFunction{
	@Override
	public FunctionValues getValues(double[] point){
		double p = point[0];
		System.out.println("Calculating function for x="+p);
		return new FunctionValues(Math.pow(p+4, 2), 
				new double[]{2*(p+4)});
	}

	@Override
	public double fun(double[] point) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void grad(double[] grad) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void Hv(double[] d, double[] Hd) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int get_nr_variable() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double fun() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void grad(int inst, double[] g) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int get_nr_instances() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void hessian(int ii, double[] hessian) {
		// TODO Auto-generated method stub
		
	}
}

//class SampleListener implements IterationFinishedListener{
//	int i = 0;
//	@Override
//	public boolean iterationFinished(double[] point,
//			double functionValue, double[] gradient) {
//		System.out.println("Iteration "+i+" finished with x="+point[0]+
//				", function value="+functionValue+", gradient="+gradient[0]);
//		i++;
//		return true;
//	}
//}
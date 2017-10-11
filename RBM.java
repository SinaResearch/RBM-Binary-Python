/**
 *
 * @modifications Sina
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.io.IOException;
import java.util.Scanner;
import java.io.*;
import java.util.Arrays;


public class RBM {
	public int N;
	public int n_visible;
	public int n_hidden;
	public double[][] W;
	public double[] hbias;
	public double[] vbias;
	public Random rng;
        
	
	public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}
	
	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;
		
		int c = 0;
		double r;
		
		for(int i=0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}
		
		return c;
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
        
        // Initialisation des variables de la classe RBM.
	public RBM(int N, int n_visible, int n_hidden, double[][] W, double[] hbias, double[] vbias, Random rng) { 
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;
	
		if(rng == null)	this.rng = new Random(1234); // dans le cas de pas instiallisation de rng c'est fait.
		else this.rng = rng;
		
		if(W == null) { // il crée un array W de deux dimentions (n_hidden x n_visible) qui contient le calcul du méthod uniform(-a, a). 
			this.W = new double[this.n_hidden][this.n_visible];
			double a = 1.0 / this.n_visible;
			
			for(int i=0; i<this.n_hidden; i++) {
				for(int j=0; j<this.n_visible; j++) {
					this.W[i][j] = uniform(-a, a); 
				}
			}	
		} else {
			this.W = W;
		}
		
		if(hbias == null) { // il crée un array hbias et met autant fois qu'on a n_hidden 0 dedans. 
			this.hbias = new double[this.n_hidden];
			for(int i=0; i<this.n_hidden; i++) this.hbias[i] = 0;
		} else {
			this.hbias = hbias;
		}
		
		if(vbias == null) { // il crée un array vbias et met autant fois qu'on a n_visible 0 dedans. 
			this.vbias = new double[this.n_visible];
			for(int i=0; i<this.n_visible; i++) this.vbias[i] = 0;
		} else {
			this.vbias = vbias;
		}
	}
		
	public void contrastive_divergence(double[] input, double lr, int k) { // lr=1000 k=1 n_hidden=3 n_visible=6 
		double[] ph_mean = new double[n_hidden];
		double[] ph_sample = new double[n_hidden];
		double[] nv_means = new double[n_visible];
		double[] nv_samples = new double[n_visible];
		double[] nh_means = new double[n_hidden];
		double[] nh_samples = new double[n_hidden];
		
		/* CD-k */
		sample_h_given_v(input, ph_mean, ph_sample);
		
		for(int step=0; step<k; step++) {
			if(step == 0) {
				gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
			} else {
				gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
			}
		}
		
		for(int i=0; i<n_hidden; i++) {
			for(int j=0; j<n_visible; j++) {
				// W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
				W[i][j] += lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
			}
			hbias[i] += lr * (ph_sample[i] - nh_means[i]) / N;
		}
		

		for(int i=0; i<n_visible; i++) {
			vbias[i] += lr * (input[i] - nv_samples[i]) / N;
		}

	}
	
        /*
        This function infers state of hidden units given visible units '''
        compute the activation of the hidden units given a sample of
        the visibles
        */
	public void sample_h_given_v(double[] v0_sample, double[] mean, double[] sample) {
		for(int i=0; i<n_hidden; i++) {
			mean[i] = propup(v0_sample, W[i], hbias[i]);
			sample[i] = binomial(1, mean[i]);
		}
	}
        /*
        This function infers state of visible units given hidden units '''
        compute the activation of the visible given the hidden sample
        */
	public void sample_v_given_h(double[] h0_sample, double[] mean, double[] sample) {
		for(int i=0; i<n_visible; i++) {
			mean[i] = propdown(h0_sample, i, vbias[i]);
			sample[i] = binomial(1, mean[i]);
		}
	}
	/*
        This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        */
	public double propup(double[] v, double[] w, double b) {
		double pre_sigmoid_activation = 0.0;
		for(int j=0; j<n_visible; j++) {
			pre_sigmoid_activation += w[j] * v[j];
		}
		pre_sigmoid_activation += b;
		return sigmoid(pre_sigmoid_activation);
	}
	/*
        This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        */
	public double propdown(double[] h, int i, double b) {
	  double pre_sigmoid_activation = 0.0;
	  for(int j=0; j<n_hidden; j++) {
	    pre_sigmoid_activation += W[j][i] * h[j];
	  }
	  pre_sigmoid_activation += b;
	  return sigmoid(pre_sigmoid_activation);
	}
	/*
        gibbs_vhv which performs a step of Gibbs sampling starting from the visible units.
        As we shall see, this will be useful for sampling from the RBM.
        This function implements one step of Gibbs sampling,
        starting from the hidden state
        */
	public void gibbs_hvh(double[] h0_sample, double[] nv_means, double[] nv_samples, double[] nh_means, double[] nh_samples) {
	  sample_v_given_h(h0_sample, nv_means, nv_samples);
	  sample_h_given_v(nv_samples, nh_means, nh_samples);
	}

	public void reconstruct(double[] v, double[] reconstructed_v) {
	  double[] h = new double[n_hidden];
	  double pre_sigmoid_activation;
	
	  for(int i=0; i<n_hidden; i++) {
	    h[i] = propup(v, W[i], hbias[i]);
	  }
	
	  for(int i=0; i<n_visible; i++) {
	    pre_sigmoid_activation = 0.0;
	    for(int j=0; j<n_hidden; j++) {
	      pre_sigmoid_activation += W[j][i] * h[j];
	    }
	    pre_sigmoid_activation += vbias[i];
	
	    reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
	  }	
	}	
	
        /* 
        RBM:
        A trained restricted Boltzmann machine will learn the structure of the data fed into it via the visible layer;
        it does so through the act of reconstructing the data again and again, with its reconstructions increasing their
        similarity to the benchmark, original data. The ever-decreasing difference between the RBM’s reconstruction and the
        benchmark is measured with a loss function. The restricted Boltzmann machine takes each step closer to the
        original using algorithms like stochastic gradient descent.
        
        Learning_rate:
        learningRate, like momentum, affects how much the neural net adjusts the coefficients 
        on each iteration as it corrects for error. These two parameters help determine the size of 
        the steps the net takes down the gradient towards a local optimum. A large learning rate will 
        make the net learn fast, and maybe overshoot the optimum. A small learning rate will slow down 
        the learning, which can be inefficient.
        
        VisibleUnit/hiddenUnit:
        visibleUnit/hiddenUnit refers to the layers of a neural net. The visible unit, or layer, is the layer 
        of nodes where input goes in, and the hiddenUnit is the layer where those inputs are recombined 
        in more complex features. Both units have their own so-called transforms, in this case Gaussian 
        for the visible and Rectified Linear for the hidden, which map the signal coming out of their respective 
        layers onto a new space.
        
        Momentum:
        Momentum is a simple method for increasing the speed of learning when the objective function
        contains long, narrow and fairly straight ravines with a gentle but consistent gradient along the floor 
        of the ravine and much steeper gradients up the sides of the ravine.
        
        K:
        The variable k is the number of times you run contrastive divergence. Each time contrastive divergence
        is run, it’s a sample of the Markov chain composing the restricted Boltzmann machine. A typical value is 1.
        */
        
	private static double[][] test_rbm(double learning_rate,int training_epochs,int k, int train_N,int test_N,int n_visible,int n_hidden
                                    ,double[][] train_X,double[][] test_X) 
        {                
		Random rng = new Random(123);                
		//initialisation
		RBM rbm = new RBM(train_N, n_visible, n_hidden, null, null, null, rng);
		// train
                // Apprentissage de l'algorithme
		for(int epoch=0; epoch<training_epochs; epoch++) {
			for(int i=0; i<train_N; i++) {
				rbm.contrastive_divergence(train_X[i], learning_rate, k);
			}
		}
		
		double[][] reconstructed_X = new double[test_N][n_visible];
                double[][] result=new double[test_N][n_visible];
                
		for(int i=0; i<test_N; i++) {
			rbm.reconstruct(test_X[i], reconstructed_X[i]);
			for(int j=0; j<n_visible; j++) {
                            result[i][j]=reconstructed_X[i][j];
                            //System.out.printf("%.5f ", reconstructed_X[i][j]);
			}
			//System.out.println();
		}
                
                return result;
	}
        
	private static void test_rbm(double learning_rate,int training_epochs,int k, int train_N,int n_visible,int n_hidden,double[][] train_X) 
        {                
		Random rng = new Random(123);                
		//initialisation
		RBM rbm = new RBM(train_N, n_visible, n_hidden, null, null, null, rng);
		// train
                // Apprentissage de l'algorithme
		for(int epoch=0; epoch<training_epochs; epoch++) {
			for(int i=0; i<train_N; i++) {
				rbm.contrastive_divergence(train_X[i], learning_rate, k);
			}
		}
	}
	
	public static void main(String[] args) {
            //--------------------------------
            Scanner trainingSet_filename_scanner = new Scanner(System.in);
            Scanner testSet_filename_scanner = new Scanner(System.in);
            Scanner epoch_scanner = new Scanner(System.in);
            Scanner hidden_unit = new Scanner(System.in);
            
            System.out.print("Training set filename: ");
            String trainingSet_filename = trainingSet_filename_scanner.nextLine(); 
            
            System.out.print("Test set filename: ");
            String testSet_filename = testSet_filename_scanner.nextLine(); 
            
            System.out.print("Number of the epochs: ");
            String epoch = epoch_scanner.nextLine();
            
            System.out.print("Number of the hidden units: ");
            int hiddenNum = Integer.parseInt(hidden_unit.nextLine());
            
            String directory=trainingSet_filename.substring(0,trainingSet_filename.lastIndexOf("\\"));
            directory=directory.substring(0,directory.lastIndexOf("\\"))+"\\RBM Yusugomori";
            
            File file = new File(directory);
            if (!file.exists()) {
                    if (!file.mkdir()) {
                            System.out.println("Failed to create directory!");
                    }
            }            
            //--------------------------------
            double learning_rate = 0.01;
            /* The learning rate, LR, applies a greater or lesser portion of the respective adjustment 
            to the old weight. If the factor is set to a large value, then the neural network may learn more quickly,
            but if there is a large variability in the input set then the network may not learn very well or at all.
            */
            int training_epochs = Integer.parseInt(epoch);; // each time the network is presented with a new input pattern
            int k = 1;

            int trainingNumber=0;
            int testNum=0;
            int visibleNum=0;
            double[][] trainingVector=new double[0][0];
            double[][] testVector=new double[0][0];
            
            String output=directory+"\\"+"Test result "+epoch+"-"+hiddenNum+" LR "+learning_rate+" "+".txt";
            
            try{  
                String[] trainingSet = (new Scanner(new File(trainingSet_filename.replace("\\","\\\\"))).useDelimiter("\\Z").next()).split("\n");
                String[] testSet = (new Scanner(new File(testSet_filename.replace("\\","\\\\"))).useDelimiter("\\Z").next()).split("\n");

                PrintWriter testResult = new PrintWriter(output, "UTF-8");
                
                trainingNumber=trainingSet.length;
                testNum=testSet.length;
                visibleNum=trainingSet[0].split(",").length;
                System.out.println(visibleNum+" visible classes and "+hiddenNum+" hidden classes found.");
                // training data
                String[] trainingValues;
                trainingVector=new double[trainingNumber][visibleNum];
                for(int i=0;i<trainingSet.length;i++){
                    trainingValues=trainingSet[i].split(",");
                    for(int j=0;j<trainingValues.length;j++){
                        trainingVector[i][j]=Double.parseDouble(trainingValues[j]);
                    }
                }
                // test data
                // Test d'un nouveau vecteur
                String[] testValues;
                testVector=new double[testSet.length][visibleNum];
                for(int z=0;z<testSet.length;z++){
                    testValues=testSet[z].split(",");
                    for(int w=0;w<testValues.length;w++){
                        testVector[z][w]=Double.parseDouble(testValues[w]);
                    }
                }                    
                
                double[][] res=test_rbm(learning_rate,training_epochs,k,trainingNumber, testNum,visibleNum,hiddenNum,trainingVector,testVector);
                
                for(int m=0;m<res.length;m++){
                    testResult.println(Arrays.toString(res[m]));
                    testResult.flush();
                }
            }
            catch(FileNotFoundException exp)
            {
               System.out.println("IOError: FileNOTFOUND"+"\n");
                exp.printStackTrace();
            }
            catch(IOException exc)
            {
              System.out.println("IOError:IOException" + exc.getMessage() + "\n");
              exc.printStackTrace();
            }

	}
	
}

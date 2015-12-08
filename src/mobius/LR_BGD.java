package mobius;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class LR_BGD {

	public enum MSECounter {
		Terminate
	}
	
	public static final float LAMBDA =0.1f;
	public static final int DIM = 7;
	public final static float THRESHOLD = 0.5f;
	public final static int MAX_ITERATION_NUM = 3;
	public final static int MAX_AGE = 45;
	
	public static final String FILESYS_SCHEMA = "s3://finalapp";
	public static final String WEIGHT_DATA_PATH = "s3://finalapp/weight/data.txt";
	public static final String AGE_DATA_PATH = "s3://finalapp/age/data.txt";
	public static final String OLD_MSE_DATA_PATH = "s3://finalapp/old/data.txt";

//	public static final String FILESYS_SCHEMA = "file:///home/unoboros/workspace1/Mobius";
//	public static final String WEIGHT_DATA_PATH = "/home/unoboros/workspace1/Mobius/weight/data.txt";
//	public static final String AGE_DATA_PATH = "/home/unoboros/workspace1/Mobius/age/data.txt";
//	public static final String OLD_MSE_DATA_PATH = "/home/unoboros/workspace1/Mobius/old/data.txt";

	public static class MSEMapper extends Mapper<Object, Text, Text, Text> {
		public float[] weights = new float[DIM];
		public float sum = 0;
		public int n = 0;
		public int age;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			try {
				FileSystem fs = FileSystem.get(new URI(FILESYS_SCHEMA),context.getConfiguration());
//				FileSystem fs = FileSystem.get(context.getConfiguration());
				FSDataInputStream in = fs.open(new Path(WEIGHT_DATA_PATH));
				BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
				String line;
				if ((line = rdr.readLine()) != null) {
					// read in weights here
					String[] ws = line.split(",");
					assert(ws.length == DIM);
					for(int i=0;i<DIM;i++){
						weights[i] = Float.parseFloat(ws[i]);
					}
				}
				rdr.close();
				in.close();
				
				in = fs.open(new Path(AGE_DATA_PATH));
				rdr = new BufferedReader(new InputStreamReader(in));
				line = rdr.readLine();
				age = Integer.parseInt(line);
				rdr.close();
				in.close();
				
			} catch (IOException e) {
				throw new RuntimeException(e);
			} 
			catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			// calculate SE here, and using a in-map combiner to get the
			// partial summation
			String strs[] = value.toString().split(",");
//			assert(strs.length == DIM + 1);
//			float x[] = new float[DIM+1];
//			for(int i=0;i<DIM+1;i++){
//				x[i] = Float.parseFloat(strs[i]);
//			}
//			float y = x[DIM];
			float x[] = new float[DIM];
			for(int i=0;i<DIM;i++){
				x[i] = Float.parseFloat(strs[i]);
			}
			float y = x[2]>=age?1:0;
			float err = (logistic(weights, x)-y);
			sum += err*err;
			n++;
		}
		
		protected void cleanup(Context context) throws IOException, InterruptedException {
			StringBuilder sb = new StringBuilder();
			sb.append(""+sum+",");
			sb.append(""+n);	
			context.write(new Text("_"), new Text(sb.toString()));
		}
	}

	public static class MSEReducer extends Reducer<Text, Text, NullWritable, Text> {

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			int n = 0;
			float sum = 0;
			for (Text value:values){
				String[] comp = value.toString().split(",");
				sum += Float.parseFloat(comp[0]);
				n += Integer.parseInt(comp[1]);
			}
			
			float mse = sum/n;
			if(mse <= THRESHOLD)
				context.getCounter(MSECounter.Terminate).increment(1L);
			
			context.write(NullWritable.get(), new Text(""+mse));
			
		}
	}

	public static class BGDMapper extends Mapper<Object, Text, Text, Text> {
		public float[] weights = new float[DIM];;
		public float[] partials = new float[DIM];
		public int age;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			try {
				FileSystem fs = FileSystem.get(new URI(FILESYS_SCHEMA),context.getConfiguration());
//				FileSystem fs = FileSystem.get(context.getConfiguration());
				FSDataInputStream in = fs.open(new Path(WEIGHT_DATA_PATH));
				BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
				String line;
				if ((line = rdr.readLine()) != null) {
					// read in weights here
					String[] ws = line.split(",");
					assert(ws.length == DIM);
					for(int i=0;i<DIM;i++){
						weights[i] = Float.parseFloat(ws[i]);
						partials[i] = 0;
					}
				}
				rdr.close();
				in.close();
				
				in = fs.open(new Path(AGE_DATA_PATH));
				rdr = new BufferedReader(new InputStreamReader(in));
				line = rdr.readLine();
				age = Integer.parseInt(line);
				rdr.close();
				in.close();
				
			} catch (IOException e) {
				throw new RuntimeException(e);
			} 
			catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			String strs[] = value.toString().split(",");
			assert(strs.length == DIM + 1);
			float x[] = new float[DIM+1];
			try{
//				for(int i=0;i<DIM+1;i++){
//					x[i] = Float.parseFloat(strs[i]);
//				}
//				float y = x[DIM];
//				for(int j=0; j < DIM; j++){
//					partials[j] += (logistic(weights, x)-y)*x[j];
//				}				
				for(int i=0;i<DIM;i++){
					x[i] = Float.parseFloat(strs[i]);
				}
				float y = x[2]>=age?1:0;
				for(int j=0; j < DIM; j++){
					partials[j] += (logistic(weights, x)-y)*x[j];
				}				

			}catch(Exception e){
				e.printStackTrace();
			}
		}
		
		protected void cleanup(Context context) throws IOException, InterruptedException {
			StringBuilder sb = new StringBuilder();
			for(float p: partials){
				sb.append(""+p+",");
			}
			context.write(new Text("_"), new Text(sb.toString()));
		}
	}

	public static class BGDReducer extends Reducer<Text, Text, NullWritable, Text> {

		FileSystem fs;
		FSDataOutputStream out;
		Path weightPath;
		public float[] weights = new float[DIM];;
		public float[] partials = new float[DIM];
		StringBuilder newWeights = new StringBuilder();

		@Override
		public void setup(Context context) throws IOException, InterruptedException {

			try {
				fs = FileSystem.get(new URI(FILESYS_SCHEMA),context.getConfiguration());
//				fs = FileSystem.get(context.getConfiguration());
				FSDataInputStream in = fs.open(new Path(WEIGHT_DATA_PATH));
				BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
				String line;
				// For each record in the user file
				if ((line = rdr.readLine()) != null) {
					String[] ws = line.split(",");
					assert(ws.length == DIM);
					for(int i=0;i<DIM;i++){
						weights[i] = Float.parseFloat(ws[i]);
						partials[i] = 0;
					}
				}
				rdr.close();
				in.close();

			} catch (IOException e) {
				throw new RuntimeException(e);
			} 
			catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			// Summation form
			for(Text value:values){
				String[] contrib = value.toString().split(",");
				assert(contrib.length == DIM);
				for(int i=0;i<DIM;i++){
					partials[i] += Float.parseFloat(contrib[i]);
				}
			}
			
			// Weights update
			for(int i=0;i<DIM;i++){
				float wi = weights[i]-LAMBDA*partials[i];
				newWeights.append(""+wi+",");
			}
		}

		protected void cleanup(Context context) throws IOException, InterruptedException {
			weightPath = new Path(WEIGHT_DATA_PATH);
			out = fs.create(weightPath);
			out.write(newWeights.toString().getBytes());
			out.close();
		}
	}


	/*
	 * Logistic function for classification
	 */
	public static float logistic(float[] weights, float[] xs) {
		assert(weights.length==xs.length-1);
		float z = 0;
		for(int i=0;i<weights.length;i++){
			z += weights[i]*xs[i];
		}
		return cap((float) (1.0/(1+Math.exp(-z))));
	}
	
	private static float cap(float x){
		return x>=0.5?1:0;
//		return x;
	}
	
	public static void main(String[] args) throws Exception {
		int count = 0;
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		float[] min_mse = new float[MAX_AGE+1];
		if (otherArgs.length < 3) {
			System.err.println("Usage: LR_BGD <data_folder> [<data_folder>...] <weight_folder> <output_folder>");
			System.exit(2);
		}
		
		for(int age=0; age<=MAX_AGE; age++){
			
//			FileSystem fs = FileSystem.get(conf);
			FileSystem fs = FileSystem.get(new URI(FILESYS_SCHEMA),conf);
			FSDataOutputStream out = fs.create(new Path(AGE_DATA_PATH));
			out.write((""+age).getBytes());
			out.close();
			
			out = fs.create(new Path(OLD_MSE_DATA_PATH));
			out.write(("1").getBytes());
			out.close();
			
			out = fs.create(new Path(WEIGHT_DATA_PATH));
			out.write("1,1,1,1,1,1,1".toString().getBytes());
			out.close();
			
			while (count < MAX_ITERATION_NUM) {
				conf = new Configuration();
				System.err.println("ROUND               "+count);
				if(round(otherArgs, count==(MAX_ITERATION_NUM-1),conf)==0){
					count++;
					continue;
				}
				break;
			}
			
			count = 0;
			
//			fs = FileSystem.get(conf);
			fs = FileSystem.get(new URI(FILESYS_SCHEMA),conf);
			FSDataInputStream in = fs.open(new Path(OLD_MSE_DATA_PATH));
			BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
			float mse = Float.parseFloat(rdr.readLine());
			rdr.close();
			in.close();
			min_mse[age] = mse;
			System.err.println("MSE = "+mse+" AGE="+age);
		}
		
		int min_age = MAX_AGE;
		float min_e = 1;
		for(int age=0; age<=MAX_AGE; age++){
			 if (min_mse[age]<min_e){
				 min_age = age;
				 min_e = min_mse[age];
			 }
		}
		
//		FileSystem fs = FileSystem.get(conf);
		FileSystem fs = FileSystem.get(new URI(FILESYS_SCHEMA),conf);
		FSDataOutputStream out = fs.create(new Path(AGE_DATA_PATH));
		out.write((""+min_age).getBytes());
		out.close();
		System.exit(0);			
	}

	private static long round(String[] otherArgs, boolean lastRound, Configuration conf)
			throws IOException, InterruptedException, ClassNotFoundException {
		/*
		 * Updating weights using Batch GD
		 */
		Job jobBGD = Job.getInstance(conf, "LR_BGD");
		jobBGD.setJarByClass(LR_BGD.class);
		jobBGD.setMapperClass(BGDMapper.class);
		jobBGD.setReducerClass(BGDReducer.class);
		jobBGD.setNumReduceTasks(1);
		jobBGD.setOutputKeyClass(Text.class);
		jobBGD.setOutputValueClass(Text.class);
		for (int i = 0; i < otherArgs.length - 2; ++i) {
			FileInputFormat.addInputPath(jobBGD, new Path(otherArgs[i]));
		}
		FileOutputFormat.setOutputPath(jobBGD, new Path(otherArgs[otherArgs.length - 1]));

		boolean finished = jobBGD.waitForCompletion(false);

		FileSystem fs;
		Path outputFolder = new Path(otherArgs[otherArgs.length - 1]);
		try {
			fs = FileSystem.get(new URI(FILESYS_SCHEMA),conf);
//			fs = FileSystem.get(conf);
			// true stands for recursively deleting the folder you gave
			if (fs.exists(outputFolder)) {
				fs.delete(outputFolder, true);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		if (!finished){			
			System.err.println("jobBGD aborted!");
			System.exit(1);
		}

		/*
		 * Calculating MSE and decide whether continue the iteration. MSEReduce
		 * task should sum the SEs and get the mean then update global indicator
		 */
		Job jobMSE = Job.getInstance(conf, "LR_MSE");
		jobMSE.setJarByClass(LR_BGD.class);
		jobMSE.setMapperClass(MSEMapper.class);
		jobMSE.setReducerClass(MSEReducer.class);
		jobMSE.setNumReduceTasks(1);
		jobMSE.setOutputKeyClass(Text.class);
		jobMSE.setOutputValueClass(Text.class);
		for (int i = 0; i < otherArgs.length - 2; ++i) {
			FileInputFormat.addInputPath(jobMSE, new Path(otherArgs[i]));
		}
		FileOutputFormat.setOutputPath(jobMSE, new Path(otherArgs[otherArgs.length - 1]));

		finished = jobMSE.waitForCompletion(false);

		if (!finished){			
			System.err.println("jobMSE aborted!");
			System.exit(1);
		}

		// checking the global indicator to decide if continue to next
		// iteration
		long terminate = jobMSE.getCounters().findCounter(MSECounter.Terminate).getValue();

		try {
			fs = FileSystem.get(new URI(FILESYS_SCHEMA),conf);
//			fs = FileSystem.get(conf);
			FSDataInputStream in = fs.open(new Path(otherArgs[otherArgs.length - 1]+"/part-r-00000"));
			BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
			float mse = Float.parseFloat(rdr.readLine());
			rdr.close();
			in.close();
			in = fs.open(new Path(OLD_MSE_DATA_PATH));
			rdr = new BufferedReader(new InputStreamReader(in));
			float old_mse = Float.parseFloat(rdr.readLine());
			rdr.close();
			in.close();
			if(mse < old_mse){
				FSDataOutputStream out = fs.create(new Path(OLD_MSE_DATA_PATH));
				out.write((""+mse).getBytes());
				out.close();
			}
			System.err.println("E = "+mse);
			fs.delete(outputFolder, true);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		return terminate;
	}

}

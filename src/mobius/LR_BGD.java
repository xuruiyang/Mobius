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
import org.apache.hadoop.io.FloatWritable;
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
	
	public static final float LAMBDA = 0.1f;
	public static final int DIM = 5;
	public final static float THRESHOLD = 0.3f;
	public final static int MAX_ITERATION_NUM = 100;

	public static class MSEMapper extends Mapper<Object, Text, Text, FloatWritable> {
		
		public float[] weights;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			try {
				FileSystem fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				FSDataInputStream in = fs.open(new Path("s3://finalapp/weight/data.txt"));
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
				
			} catch (IOException e) {
				throw new RuntimeException(e);
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			// calculate SE here, and using a in-map combiner to get the
			// partial summation
			String strs[] = value.toString().split(",");
			assert(strs.length == DIM + 1);
			float x[] = new float[DIM+1];
			for(int i=0;i<DIM+1;i++){
				x[i] = Float.parseFloat(strs[i]);
			}
			float y = x[DIM];
			float err = (logistic(weights, x)-y);
			context.write(new Text("_"), new FloatWritable(err*err));
		}
	}

	public static class MSEReducer extends Reducer<Text, FloatWritable, NullWritable, Text> {

		public void reduce(Text key, Iterable<FloatWritable> values, Context context)
				throws IOException, InterruptedException {
			
			int n = 0;
			float sum = 0;
			for (FloatWritable value:values){
				float v = value.get();
				sum += v;
				n++;
			}
			
			float mse = sum/n;
			if(mse <= THRESHOLD)
				context.getCounter(MSECounter.Terminate).increment(1);
			
			context.write(NullWritable.get(), new Text("MSE = "+mse));
			
		}
	}

	public static class BGDMapper extends Mapper<Object, Text, Text, Text> {
		public float[] weights;
		public float[] partials = new float[DIM];

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			try {
				FileSystem fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				FSDataInputStream in = fs.open(new Path("s3://finalapp/weight/data.txt"));
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
				
			} catch (IOException e) {
				throw new RuntimeException(e);
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			String strs[] = value.toString().split(",");
			assert(strs.length == DIM + 1);
			float x[] = new float[DIM+1];
			for(int i=0;i<DIM+1;i++){
				x[i] = Float.parseFloat(strs[i]);
			}
			float y = x[DIM];
			for(int j=0; j < DIM; j++){
				partials[j] += (logistic(weights, x)-y)*x[j];
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
		public float[] weights;
		public float[] partials = new float[DIM];
		StringBuilder newWeights = new StringBuilder();

		@Override
		public void setup(Context context) throws IOException, InterruptedException {

			try {
				fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				FSDataInputStream in = fs.open(new Path("s3://finalapp/weight/data.txt"));
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
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			// Summation form here
			for(Text value:values){
				String[] contrib = value.toString().split(",");
				assert(contrib.length == DIM);
				for(int i=0;i<DIM;i++){
					partials[i] += Float.parseFloat(contrib[i]);
				}
			}
			
			for(int i=0;i<DIM;i++){
				float wi = weights[i]-LAMBDA*partials[i];
				newWeights.append(""+wi+",");
			}
		}

		protected void cleanup(Context context) throws IOException, InterruptedException {
			weightPath = new Path("s3://finalapp/weight/data.txt");
			out = fs.create(weightPath);
			out.write(newWeights.toString().getBytes());
			out.close();
		}
	}

	public static void main(String[] args) throws Exception {
		int count = 0;
		while (count < MAX_ITERATION_NUM) {
			if(round(args)==0){
				count++;
				continue;
			}
			break;
		}
		
		if(count==MAX_ITERATION_NUM){			
			System.err.println("Unable to converge!");
			System.exit(1);
		}
		System.exit(0);
	}

	public static float logistic(float[] weights, float[] xs) {
		assert(weights.length==xs.length-1);
		float z = 0;
		for(int i=0;i<weights.length;i++){
			z += weights[i]*xs[i];
		}
		return (float) (1.0/(1+Math.exp(-z)));
	}

	private static long round(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 3) {
			System.err.println("Usage: LR_BGD <data_folder> <weight_folder> <output_folder>");
			System.exit(2);
		}

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
		FileInputFormat.addInputPath(jobBGD, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(jobBGD, new Path(otherArgs[2]));

		boolean finished = jobBGD.waitForCompletion(true);

		// configuration should contain reference to your namenode
		FileSystem fs = FileSystem.get(conf);
		// true stands for recursively deleting the folder you gave
		Path outputFolder = new Path(otherArgs[2]);
		if (fs.exists(outputFolder)) {
			fs.delete(outputFolder, true);
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
		jobMSE.setOutputValueClass(FloatWritable.class);
		FileInputFormat.addInputPath(jobMSE, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(jobMSE, new Path(otherArgs[2]));

		jobMSE.getCounters().findCounter(MSECounter.Terminate).setValue(0);

		finished = jobMSE.waitForCompletion(true);

		if (!finished){			
			System.err.println("jobMSE aborted!");
			System.exit(1);
		}

		// checking the global indicator to decide if continue to next
		// iteration
		long terminate = jobMSE.getCounters().findCounter(MSECounter.Terminate).getValue();

		if (terminate==0 && fs.exists(outputFolder)) {
			// reserve the last output result
			fs.delete(outputFolder, true);
		}
		return terminate;
	}

}

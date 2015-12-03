package mobius;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
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

@SuppressWarnings("deprecation")
public class LR_BGD {

	public enum MSECounter {
		Terminate
	}
	
	public static final float LAMBDA = 0.1f;
	public static final int DIM = 5;
	public final static float THRESHOLD = 0.3f;
	public final static int MAX_ITERATION_NUM = 100;

	public static class MSEMapper extends Mapper<Object, Text, Text, FloatWritable> {
		
		public String[] weights;
		public float sum; 

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			sum = 0;
			try {
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("File information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					Path p = new Path(u);
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File(p.toString()))));

					String line;
					// For each record in the user file
					if ((line = rdr.readLine()) != null) {
						// read in weights here
						weights = line.split(",");
						assert(weights.length == DIM);
					}
					
					rdr.close();
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			// calculate SE here, and using a in-map combiner to get the
			// partial summation
			String strs[] = value.toString().split(",");
			assert(strs.length == DIM + 1);
			float y = Float.parseFloat(strs[DIM]);
			float err = (logistic(weights, strs)-y);
			sum+= err*err;
			context.write(new Text("_"), new FloatWritable(sum));
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
		public String[] weights;
		public String[] partials = new String[DIM];

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			for(int i=0;i<DIM;i++){
				partials[i] = "0";
			}
			
			try {
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("File information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					Path p = new Path(u);
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File(p.toString()))));

					String line;
					// For each record in the user file
					if ((line = rdr.readLine()) != null) {
						// read in weights here
						weights = line.split(",");
						assert(weights.length == DIM);
					}
					
					rdr.close();
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			String strs[] = value.toString().split(",");
			assert(strs.length == DIM + 1);
			for(int j=0; j < DIM; j++){
				float xj = Float.parseFloat(strs[j]);
				float y = Float.parseFloat(strs[DIM]);
				float partial = (logistic(weights, strs)-y)*xj + Float.parseFloat(partials[j]);
				partials[j] = ""+partial;
			}
		}
		
		protected void cleanup(Context context) throws IOException, InterruptedException {
			StringBuilder sb = new StringBuilder();
			for(String s: partials){
				sb.append(s+",");
			}
			context.write(new Text("_"), new Text(sb.toString()));
		}
	}

	public static class BGDReducer extends Reducer<Text, Text, NullWritable, Text> {

		FileSystem fs;
		FSDataOutputStream out;
		Path weightPath;
		public String[] weights;
		public String[] partials = new String[DIM];

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			
			for(int i=0;i<DIM;i++){
				partials[i] = "0";
			}

			fs = FileSystem.get(context.getConfiguration());
			weightPath = new Path("weight/data.txt");
			if (fs.exists(weightPath)) {
				fs.delete(weightPath);
			}
			out = fs.create(weightPath);
			try {
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("File information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					Path p = new Path(u);
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File(p.toString()))));

					String line;
					// For each record in the user file
					if ((line = rdr.readLine()) != null) {
						weights = line.split(",");
						assert(weights.length == DIM);
					}
					
					rdr.close();
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			// Summation form here
			for(Text value:values){
				String[] contrib = value.toString().split(",");
				assert(contrib.length == DIM);
				for(int i=0;i<DIM;i++){
					partials[i] = ""+(Float.parseFloat(contrib[i]) + Float.parseFloat(partials[i]));
				}
			}
			
			StringBuilder newWeights = new StringBuilder();
			for(int i=0;i<DIM;i++){
				float wi = Float.parseFloat(weights[i])-LAMBDA*Float.parseFloat(partials[i]);
				newWeights.append(""+wi+",");
			}

			out.write(newWeights.toString().getBytes());
		}

		protected void cleanup(Context context) throws IOException, InterruptedException {
			out.close();
			fs.close();
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

	public static float logistic(String[] weights, String[] strs) {
		assert(weights.length==strs.length-1);
		float z = 0;
		for(int i=0;i<weights.length;i++){
			z += Float.parseFloat(weights[i])*Float.parseFloat(strs[i]);
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
		Job jobBGD = new Job(conf, "LR_BGD");
		jobBGD.setJarByClass(LR_BGD.class);
		jobBGD.setMapperClass(BGDMapper.class);
		jobBGD.setReducerClass(BGDReducer.class);
		jobBGD.setNumReduceTasks(1);
		jobBGD.setOutputKeyClass(Text.class);
		jobBGD.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(jobBGD, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(jobBGD, new Path(otherArgs[2]));

		// Configure the DistributedCache for weights file
		jobBGD.addCacheFile(new Path(otherArgs[1] + "/data.txt").toUri());

		boolean finished = jobBGD.waitForCompletion(true);

		// configuration should contain reference to your namenode
		FileSystem fs = FileSystem.get(conf);
		// true stands for recursively deleting the folder you gave
		Path outputFolder = new Path(otherArgs[2]);
		if (fs.exists(outputFolder)) {
			fs.delete(outputFolder, true);
		}

		if (!finished)
			System.exit(1);

		/*
		 * Calculating MSE and decide whether continue the iteration. MSEReduce
		 * task should sum the SEs and get the mean then update global indicator
		 */
		Job jobMSE = new Job(conf, "LR_MSE");
		jobMSE.setJarByClass(LR_BGD.class);
		jobMSE.setMapperClass(MSEMapper.class);
		jobMSE.setReducerClass(MSEReducer.class);
		jobMSE.setNumReduceTasks(1);
		jobMSE.setOutputKeyClass(Text.class);
		jobMSE.setOutputValueClass(FloatWritable.class);
		FileInputFormat.addInputPath(jobMSE, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(jobMSE, new Path(otherArgs[2]));

		// Configure the DistributedCache for weights file
		jobMSE.addCacheFile(new Path(otherArgs[1] + "/data.txt").toUri());
		jobMSE.getCounters().findCounter(MSECounter.Terminate).setValue(0);

		finished = jobMSE.waitForCompletion(true);

		if (!finished)
			System.exit(1);

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

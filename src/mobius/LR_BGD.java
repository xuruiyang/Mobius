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


public class LR_BGD {

	public static class BGDMapper extends Mapper<Object, Text, Text, FloatWritable> {
		public float test;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			try {
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("User information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					Path p = new Path(u);
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File(p.toString()))));
					
					String line;
					// For each record in the user file
					while ((line = rdr.readLine()) != null) {
						// TODO: read in weights here
						test = Float.parseFloat(line);
					}
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		public void map(Object line, Text value, Context context) throws IOException, InterruptedException {
			float summation = 1;
			summation += test;
			context.write(new Text("_"), new FloatWritable(summation));
		}
	}

	public static class BGDReducer extends Reducer<Text, FloatWritable, NullWritable, Text> {

		FileSystem fs;
		FSDataOutputStream out;
		Path weightPath;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {

			fs = FileSystem.get(context.getConfiguration());
			weightPath = new Path("weight/data.txt");
			if (fs.exists(weightPath)) {
				fs.delete(weightPath);
			}
			out = fs.create(weightPath);
			try {
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("User information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					Path p = new Path(u);
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File(p.toString()))));

					String line;
					// For each record in the user file
					while ((line = rdr.readLine()) != null) {
						// TODO: read in weights here
					}
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		public void reduce(Text key, Iterable<FloatWritable> values, Context context)
				throws IOException, InterruptedException {
			float v = values.iterator().next().get();
			String s = Float.toString(v);
			out.write(s.getBytes());
		}

		protected void cleanup(Context context) throws IOException, InterruptedException {
			out.close();
			fs.close();
		}
	}

	public static void main(String[] args) throws Exception {
		for (int i=0;i<5;i++){
			round(args);
		}
	}

	private static void round(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 3) {
			System.err.println("Usage: LR_BGD <data_folder> <weight_folder> <output_folder>");
			System.exit(2);
		}
		Job job = new Job(conf, "LR_BGD");
		job.setJarByClass(LR_BGD.class);
		job.setMapperClass(BGDMapper.class);
		job.setReducerClass(BGDReducer.class);
		job.setNumReduceTasks(1);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(FloatWritable.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));

		// Configure the DistributedCache for weights file
		job.addCacheFile(new Path(otherArgs[1]+"/data.txt").toUri());

		boolean finished = job.waitForCompletion(true);

		// configuration should contain reference to your namenode
		FileSystem fs = FileSystem.get(conf);
		// true stands for recursively deleting the folder you gave
		Path outputFolder = new Path(otherArgs[2]);
		if (fs.exists(outputFolder)) {
			fs.delete(outputFolder, true);
		}

		if(!finished)
			System.exit(finished ? 0 : 1);
	}

}

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

public class FileSysTest {
	public static class BGDMapper extends Mapper<Object, Text, Text, FloatWritable> {
		public float test;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			try {
				FileSystem fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				FSDataInputStream in = fs.open(new Path("s3://finalapp/weight/data.txt"));
				BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
				String line;
				while ((line = rdr.readLine()) != null) {
					// TODO: read in weights here
					test = Float.parseFloat(line);
				}
				in.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			} catch (URISyntaxException e) {
				e.printStackTrace();
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
		String s;
		float test;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			try {
				FileSystem fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				FSDataInputStream in = fs.open(new Path("s3://finalapp/weight/data.txt"));
				BufferedReader rdr = new BufferedReader(new InputStreamReader(in));
				String line;
				while ((line = rdr.readLine()) != null) {
					// TODO: read in weights here
					test = Float.parseFloat(line);
				}
				in.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			} catch (URISyntaxException e) {
				e.printStackTrace();
			}
		}

		public void reduce(Text key, Iterable<FloatWritable> values, Context context)
				throws IOException, InterruptedException {
			float v = values.iterator().next().get();
			s = Float.toString(v);
		}

		protected void cleanup(Context context) throws IOException, InterruptedException {
			try {
				fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				weightPath = new Path("s3://finalapp/weight/data.txt");
				out = fs.create(weightPath);
				out.write(s.getBytes());
				out.close();
			} catch (URISyntaxException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public static void main(String[] args) throws Exception {
		for (int i = 0; i < 5; i++) {
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
		Job job = Job.getInstance(conf, "LR_BGD");
		job.setJarByClass(FileSysTest.class);
		job.setMapperClass(BGDMapper.class);
		job.setReducerClass(BGDReducer.class);
		job.setNumReduceTasks(1);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(FloatWritable.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));

		boolean finished = job.waitForCompletion(true);
		// configuration should contain reference to your namenode
		try {
			FileSystem fs = FileSystem.get(new URI("s3://finalapp"),conf);
			// true stands for recursively deleting the folder you gave
			Path outputFolder = new Path(otherArgs[2]);
			if (fs.exists(outputFolder)) {
				fs.delete(outputFolder, true);
			}
			
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}

		if (!finished){			
			System.err.println("Aborted!");
			System.exit(finished ? 0 : 1);
		}
	}
}
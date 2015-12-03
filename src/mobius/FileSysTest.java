package mobius;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.MRJobConfig;
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
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("User information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File("./data.txt"))));
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
		String s;

		@Override
		public void setup(Context context) throws IOException, InterruptedException {
			try {
				URI[] files = context.getCacheFiles();
				if (files == null || files.length == 0) {
					throw new RuntimeException("User information is not set in DistributedCache");
				}
				// Read all files in the DistributedCache
				for (URI u : files) {
					BufferedReader rdr = new BufferedReader(
							new InputStreamReader(new FileInputStream(new File("./data.txt"))));
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
			s = Float.toString(v);
		}

		protected void cleanup(Context context) throws IOException, InterruptedException {
			try {
				fs = FileSystem.get(new URI("s3://finalapp"),context.getConfiguration());
				weightPath = new Path("s3://finalapp/tmp/data.txt");
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
			round(args, i>0);
		}
	}

	private static void round(String[] args, boolean needTimestamp) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 3) {
			System.err.println("Usage: LR_BGD <data_folder> <weight_folder> <output_folder>");
			System.exit(2);
		}
		Job job = new Job(conf, "LR_BGD");
		job.setJarByClass(FileSysTest.class);
		job.setMapperClass(BGDMapper.class);
		job.setReducerClass(BGDReducer.class);
		job.setNumReduceTasks(1);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(FloatWritable.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
		// Configure the DistributedCache for weights file
		try {
			job.addCacheFile(new URI(otherArgs[1] + "/data.txt"+"#data.txt"));
			if(needTimestamp){
				String[] timestamps = job.getFileTimestamps();
				String latestTime = timestamps[timestamps.length-1];
				StringBuilder sb = new StringBuilder();
				for(int i = 0;i<timestamps.length;i++)
					sb.append(latestTime+",");
				conf.set(MRJobConfig.CACHE_ARCHIVES_TIMESTAMPS, sb.toString());
			}
		} catch (URISyntaxException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		boolean finished = job.waitForCompletion(true);
		// configuration should contain reference to your namenode
		try {
			FileSystem fs = FileSystem.get(new URI("s3://finalapp"),conf);
			// true stands for recursively deleting the folder you gave
			Path outputFolder = new Path(otherArgs[2]);
			if (fs.exists(outputFolder)) {
				fs.delete(outputFolder, true);
			}
			
			FSDataInputStream in = fs.open(new Path("s3://finalapp/tmp/data.txt"));
			FSDataOutputStream out = fs.create(new Path("s3://finalapp/weight/data.txt"));
			byte [] b = new byte[1024];
			int len;
			while((len = in.read(b))!=-1){
				out.write(b, 0, len);
			}
			in.close();
			out.close();
			
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
		String[] strs = job.getFileTimestamps();
		StringBuilder sb = new StringBuilder();
		for(String s : strs)
			sb.append(s+" , ");
		System.err.println(sb.toString());
		if (!finished){			
			System.err.println("Aborted!");
			System.exit(finished ? 0 : 1);
		}
	}
}
package comparitor;

import java.io.File;
import java.nio.file.Paths;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import classifier.TFModels;
import general.DirectoryMethods;

public class ImageComparison {
	
	

	public static void main(String[] args) {
	    //String modelDir = TFModels.getMobilenetDirectory(); 	      
	    //String pbName = TFModels.getMobilenetV1_1_PBName();
	    
	    String modelDir = TFModels.getInception_v3_Directory();
	    String pbName = TFModels.getInception_v3_PBName();
	    
	        
	    String imagesDir = "C:\\Users\\" + TFModels.user + "\\Desktop\\Alexie\\";
	    
	    File folder = new File("C:\\Users\\299490\\Documents\\TensorFlowProject\\tf_files\\images\\logos\\pepsi");
	    String[] files = folder.list();
	    
	    String subject = imagesDir + "Subject.jpg";
	    
	    
	    for(int i=0; i<files.length; i++) {
	    	
	    }
	    /*
	    
	    String scene =  imagesDir +"\\pepsi\\pepsi1.jpg"; //"\\ipod\\117_0002.jpg";
	    
	    byte[] im1 = DirectoryMethods.readAllBytesOrExit(Paths.get(subject));
	    byte[] im2 = DirectoryMethods.readAllBytesOrExit(Paths.get(scene));
	    
	    
		byte[] graphDef = DirectoryMethods.readAllBytesOrExit(Paths.get(modelDir, pbName));
		
  	  //inception-2015-12-05 'Cast' Tensor expects decoded JPEG with 'type': 'DT_FLOAT'
		Tensor ts1 = constructAndExecuteGraphForInception_v3_2015(im1);
		Tensor ts2 = constructAndExecuteGraphForInception_v3_2015(im2);
		
		float[] kp1 = getKeypoints(graphDef, ts1);
		float[] kp2 = getKeypoints(graphDef, ts2);
		
		
		//System.out.println(cosineSimilarity(kp1,kp2));
		System.out.println(euclideanDistance(kp1, kp2));
		*/
	}
	
	private static float[] getKeypoints(byte[] graphDef, Tensor image) {
	    try (Graph g = new Graph()) {
		      g.importGraphDef(graphDef);
		      
		      
		      try (Session s = new Session(g)) {
		    	  		    	 
		    	  
		    	  Tensor result = s.runner().feed("Cast", image).fetch("pool_3").run().get(0); //retreive layer 'pool_3' 2048 keypoints
		    	  long rshape[] = result.shape();
		    	  if(result.numDimensions() != 4 || rshape[0] != 1) return null;
		    	  
		    	  //float[][][][] kps = new float[1][1][1][(int) rshape[3]];
		    	  //float[] t = result.copyTo(kps)[0][0][0];
		    	  
		    	  return result.copyTo(new float[1][1][1][(int)rshape[3]])[0][0][0];		    	  
		    	  	    	  
		      }
	    }		
	}
	
	private static Tensor constructAndExecuteGraphForInception_v3_2015(byte[] imageBytes) {
	    try (Graph g = new Graph()) {
		      Builder.GraphBuilder b = new Builder.GraphBuilder(g);
		      final int H = 224;
		      final int W = 224;
		      final float mean = 0f;
		      final float scale = 255f;

		      // Since the graph is being constructed once per execution here, we can use a constant for the
		      // input image. If the graph were to be re-used for multiple input images, a placeholder would
		      // have been more appropriate.
		      //final Output input = b.constant("input", imageBytes);
		     
		      //play with the placeholder idea
		      Output input = b.placeholder("input", DataType.STRING);
		      
		     final Output output =  b.cast(b.decodeJpeg(input, 3), DataType.FLOAT);
		     
		     		     
		      try (Session s = new Session(g)) {
		       return s.runner().feed(input, Tensor.create(imageBytes)).fetch(output.op().name()).run().get(0);
		       
		    	  
		      }
		    }		
		
	}
	

}

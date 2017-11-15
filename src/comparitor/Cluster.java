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
import general.Similarity;

public class Cluster {
	
	
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    final static int H = 224;
    final static int W = 224;
    final static float mean = 0f;
    final static float scale = 255f;
    
    
	public static void main(String[] args) {
	    //String modelDir = TFModels.getInception_v3_Directory();
	    //String pbName = TFModels.getInception_v3_PBName();
		
	    String modelDir = TFModels.getMobilenetDirectory(); 
	    String pbName = TFModels.getMobilenetV1_1_PBName("retrained_graph");
	    
	        
	    String imageDir = "C:\\Users\\" + TFModels.user + "\\Desktop\\Alexie\\";
	    
	    String imagesFolder = "C:\\Users\\299490\\Documents\\TensorFlowProject\\tf_files\\images\\logos\\pepsi\\";
	    File folder = new File(imagesFolder);
	    String[] files = folder.list();
	    
	    String imageName = "pepsi52_cropped.jpg"; 
	    String subject = imageDir + imageName;
	    
	    byte[] graphDef = DirectoryMethods.readAllBytesOrExit(Paths.get(modelDir, pbName));
	    
	    
	    
	    try (Graph inception = new Graph()) {
		      inception.importGraphDef(graphDef);
		      
		      try (Session s = new Session(inception)) {
		    	  
		    	  //create new graph to decode jpeg
		    	  Graph decoderGraph = new Graph();
		    	  Builder.GraphBuilder b = new Builder.GraphBuilder(decoderGraph);
		    	  
		    	  Session decoder = new Session(decoderGraph);
		    	  
			      Output input = b.placeholder("input", DataType.STRING);
			      
			      
				  //Output output =  b.cast(b.decodeJpeg(input, 3), DataType.FLOAT);	//for use with inceptions Cast Op
				  Output output = b.div(
			              b.sub(
				                  b.resizeBilinear(
				                      b.expandDims(
				                          b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
				                          b.constant("make_batch", 0)),
				                      b.constant("size", new int[] {H, W})),
				                  b.constant("mean", mean)),
				              b.constant("scale", scale));				      
				  
		    	  //float[] subjectKp = getKeypointsFromInception_v3(s, decoder.runner().feed(input, Tensor.create(DirectoryMethods.readAllBytesOrExit(Paths.get(subject))))
				//	    	.fetch(output.op().name()).run().get(0));	
				  
				  float[] subjectKp = getKeypointsFromMobilenet(s, decoder.runner().feed(input, Tensor.create(DirectoryMethods.readAllBytesOrExit(Paths.get(subject))))
					    	.fetch(output.op().name()).run().get(0));	
				  
				 /* loop through folder and compute similarites to subject key points */
			    for(int i=0; i<files.length; i++) {
			    	Tensor decodedJpeg = decoder.runner().feed(input, Tensor.create(DirectoryMethods.readAllBytesOrExit(Paths.get(imagesFolder + files[i]))))
			    	.fetch(output.op().name()).run().get(0);
			    	
			    	
			    	double sim = Similarity.euclideanDistance(subjectKp, getKeypointsFromMobilenet(s, decodedJpeg));//Similarity.cosineSimilarity(subjectKp, getKeypointsFromInception_v3(s, decodedJpeg)); 
			    	//System.out.println(getKeypointsFromMobilenet(s, decodedJpeg)[1]);
			    	
			    	System.out.println("Similarity between " + imageName + " and " + files[i] + " : " + sim);
			    	decodedJpeg.close();
			    }	
			    
			    decoder.close();
			    
		      } //end session
	    }
	}
	
	
	/**
	 * 
	 * @param s - Tensor session
	 * @param image - Tensor decoded image
	 * @return - 2048 key points from inception_v3 pool:3 layer
	 */
	private static float[] getKeypointsFromInception_v3(Session s, Tensor image){
	  Tensor result = s.runner().feed("Cast", image).fetch("pool_3").run().get(0); //retreive layer 'pool_3' 2048 keypoints
  	  long rshape[] = result.shape();
  	  if(result.numDimensions() != 4 || rshape[0] != 1) return null;
  	  
  	  return result.copyTo(new float[1][1][1][(int)rshape[3]])[0][0][0];			
	}
	
	private static float[] getKeypointsFromMobilenet(Session s, Tensor image) {
		  Tensor result = s.runner().feed("input", image).fetch("input_1/BottleneckInputPlaceholder").run().get(0); //retreive layer 'pool_3' 2048 keypoints
	  	  long rshape[] = result.shape();
	  	  if(result.numDimensions() != 2 || rshape[0] != 1) return null;
	  	 
	  	  return result.copyTo(new float[1][(int)rshape[1]])[0];		
		
	}
}

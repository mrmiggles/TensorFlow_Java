package classifier;

import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import general.DirectoryMethods;

public class ImageClassifier {
	
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    final static int H = 224;
    final static int W = 224;
    final static float mean = 0f;
    final static float scale = 255f;
    
    
	public static void main(String[] args) {
		classify();
	}    
	
	public static void classify(){
		
	    String modelDir = TFModels.getMobilenetDirectory(); 
	    //String pbName = TFModels.getMobilenetV1_050_PBName(); 
	    //String labelFile = TFModels.getMobilenetV1_050_Lables(); 
	      
	    String pbName = TFModels.getMobilenetV1_1_PBName("retrained_graph");
	    String labelFile = TFModels.getMobilenetV1_1_Lables("retrained_labels"); 
	    List<String> labels =
	    		DirectoryMethods.readAllLinesOrExit(Paths.get(modelDir, labelFile));	    
		    
		        
	    String imageDir = "C:\\Users\\" + TFModels.user + "\\Desktop\\Alexie\\";
	    
	    String imagesFolder = "C:\\Users\\299490\\Documents\\TensorFlowProject\\tf_files\\images\\logos\\pepsi\\";
	    File folder = new File(imagesFolder);
	    String[] files = folder.list();
	    
	    String imageName = "Subject.jpg"; 
	    String subject = imageDir + imageName;
	    
	    byte[] graphDef = DirectoryMethods.readAllBytesOrExit(Paths.get(modelDir, pbName));
	    String outputLayer = "final_result"; //"MobilenetV1/Predictions/Reshape_1";
	    
	    
	    try (Graph pretrained = new Graph()) {
		      pretrained.importGraphDef(graphDef);
		      
		      try (Session s = new Session(pretrained)) {
		    	  
		    	  //create new graph to decode jpeg
		    	  Graph decoderGraph = new Graph();
		    	  Builder.GraphBuilder b = new Builder.GraphBuilder(decoderGraph);
		    	  
		    	  Session decoder = new Session(decoderGraph);
		    	  
			      Output input = b.placeholder("input", DataType.STRING);
				  Output output = b.div(
			              b.sub(
				                  b.resizeBilinear(
				                      b.expandDims(
				                          b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
				                          b.constant("make_batch", 0)),
				                      b.constant("size", new int[] {H, W})),
				                  b.constant("mean", mean)),
				              b.constant("scale", scale));	
				  
		  
				  
				 /* loop through folder and compute similarites to subject key points */
			    for(int i=0; i<files.length; i++) {
			    	Tensor decodedJpeg = decoder.runner().feed(input, Tensor.create(DirectoryMethods.readAllBytesOrExit(Paths.get(imagesFolder + files[i]))))
			    	.fetch(output.op().name()).run().get(0);
			    	 float[] labelProbabilities = retreiveLables(s, decodedJpeg, outputLayer);
				      int bestLabelIdx = general.Executor.maxIndex(labelProbabilities);
				      System.out.println(
				         String.format(
				           files[i] + " BEST MATCH: %s (%.2f%% likely)",
				           labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
		    	
			    	decodedJpeg.close();
			    }	
			    
			    decoder.close();
			    
		      } //end session
	    }		
	}
	
	public static float[] retreiveLables(Session s, Tensor image, String outputLayerName){
		  Tensor result = s.runner().feed("input", image).fetch(outputLayerName).run().get(0);
	    	  
	        final long[] rshape = result.shape();
	        if (result.numDimensions() != 2 || rshape[0] != 1) {
	          throw new RuntimeException(
	              String.format(
	                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
	                  Arrays.toString(rshape)));
	        }
	        int nlabels = (int) rshape[1];
	        return result.copyTo(new float[1][nlabels])[0];		
	}

	
	public static void testClassification() {
		//String modelDir = TFModels.getInceptionDirectory();
	    //String pbName = TFModels.getInceptionPBName();
	    //String labelFile = TFModels.getInceptionLables();
	    
	    String modelDir = TFModels.getMobilenetDirectory(); 
	    //String pbName = TFModels.getMobilenetV1_050_PBName(); 
	    //String labelFile = TFModels.getMobilenetV1_050_Lables(); 
	      
	    String pbName = TFModels.getMobilenetV1_1_PBName();
	    String labelFile = TFModels.getMobilenetV1_1_Lables(); 
	    
	    String outputLayer = "MobilenetV1/Predictions/Reshape_1";
	    
	    String imageDir = "C:\\Users\\" + TFModels.user + "\\Desktop\\Alexie\\";
	    String imageFile = imageDir + "Subject.jpg";
	    byte[] imageBytes = DirectoryMethods.readAllBytesOrExit(Paths.get(imageFile));
	    
	    
	    byte[] graphDef = DirectoryMethods.readAllBytesOrExit(Paths.get(modelDir, pbName));
	    List<String> labels =
	    		DirectoryMethods.readAllLinesOrExit(Paths.get(modelDir, labelFile));


	    try (Tensor image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
	      float[] labelProbabilities = general.Executor.executeInceptionGraph(graphDef, image, outputLayer);
	      int bestLabelIdx = general.Executor.maxIndex(labelProbabilities);
	      System.out.println(
	         String.format(
	           "BEST MATCH: %s (%.2f%% likely)",
	           labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
	    }		
	}
	
  private static Tensor constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
	    try (Graph g = new Graph()) {
	      Builder.GraphBuilder b = new Builder.GraphBuilder(g);
	      // Some constants specific to the pre-trained model at:
	      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
	      //


	      // Since the graph is being constructed once per execution here, we can use a constant for the
	      // input image. If the graph were to be re-used for multiple input images, a placeholder would
	      // have been more appropriate.
	      final Output input = b.constant("input", imageBytes);
	      final Output output =
	          b.div(
	              b.sub(
	                  b.resizeBilinear(
	                      b.expandDims(
	                          b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
	                          b.constant("make_batch", 0)),
	                      b.constant("size", new int[] {H, W})),
	                  b.constant("mean", mean)),
	              b.constant("scale", scale));
	      	System.out.println(output.op().name());
	      try (Session s = new Session(g)) {
	        return s.runner().fetch(output.op().name()).run().get(0);
	      }
	    }
	  }	

}

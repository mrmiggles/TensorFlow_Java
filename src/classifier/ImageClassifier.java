package classifier;

import java.nio.file.Paths;
import java.util.List;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import general.DirectoryMethods;

public class ImageClassifier {

	public static void main(String[] args) {

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
	      // - The model was trained with images scaled to 224x224 pixels.
	      // - The colors, represented as R, G, B in 1-byte each were converted to
	      //   float using (value - Mean)/Scale.
	      final int H = 224;
	      final int W = 224;
	      final float mean = 0f;
	      final float scale = 255f;

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

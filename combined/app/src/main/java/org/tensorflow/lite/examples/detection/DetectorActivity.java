/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Size;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.w3c.dom.Text;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Model depthModel;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap depthCroppedFrame = null;
  private Matrix frameToDepthTransform;
  private List<Classifier.Recognition> detections;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    depthModel = new ModelFactory(getApplicationContext()).getModel(0);
    depthCroppedFrame =
            Bitmap.createBitmap(640,448, Config.ARGB_8888);
    frameToDepthTransform = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            640,448,
            sensorOrientation, MAINTAIN_ASPECT);

  }

  @Override
  public boolean onTouchEvent(MotionEvent e) {
    switch (e.getActionMasked()) {
      case (MotionEvent.ACTION_DOWN):
        tts.speak("Test", TextToSpeech.QUEUE_ADD, null, "Test");
        if (croppedBitmap != null) {
//          frameToDepthTransform
          final Canvas canvas = new Canvas(depthCroppedFrame);
          canvas.drawBitmap(rgbFrameBitmap, frameToDepthTransform, null);
          float[] pixels = getPixelFromBitmap(depthCroppedFrame);
          float[] inf = doInference(pixels);

        }
        return true;
      default:
        return true;
    }
  }

  private float[] doInference(float[] input){
    return depthModel.doInference(input, 640,448);
//    int[] coloredInference = colorMapper.applyColorMap(inference, NUMBER_THREADS);
//    outputDisp.setPixels(coloredInference, 0, resolution.getWidth(), 0, 0, resolution.getWidth(), resolution.getHeight());
//    outputDispResized = Bitmap.createScaledBitmap(outputDisp,  halfScreenSize.getWidth(), halfScreenSize.getHeight(), false);
//    outputRGB = Bitmap.createScaledBitmap(croppedFrame,  halfScreenSize.getWidth(), halfScreenSize.getHeight(), false);
  }


//  @Override
//  public void onClick(View v) {
//    super.onClick(v);
//    tts.speak("Test", TextToSpeech.QUEUE_ADD, null, "Test");
//  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);

                initiateTextToSpeech(location, result.getTitle(), new Size(300, 300));
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  public static float[] getPixelFromBitmap(Bitmap frame){
    int numberOfPixels = frame.getWidth()*frame.getHeight()*3;
    int[] pixels = new int[frame.getWidth()*frame.getHeight()];
    frame.getPixels(pixels, 0, frame.getWidth(), 0, 0, frame.getWidth(), frame.getHeight());

    float[] output = new float[numberOfPixels];

    int i = 0;
    for (int pixel : pixels) {
      output[i * 3] = Color.red(pixel) /(float)255.;
      output[i * 3 + 1] = Color.green(pixel) / (float)255.;
      output[i * 3+2] = Color.blue(pixel) / (float)255.;
      i+=1;
    }
    return output;
  }

//  private void doDepthInference
  private void initiateTextToSpeech(RectF location, String objectName, Size screenSize) {
    TextToSpeech tts = new TextToSpeech(this, null);
    tts.setLanguage(Locale.US);
    System.out.println(location);
   // int objectWidth = (int) Math.abs(location.right - location.left);
   // int objectHeight = (int) Math.abs(location.top - location.bottom);
   // if (objectWidth <= screenSize.width / 3) {  // evaluate it normally
   //   if (location.right <= screenSize.width / 3 && location.right >= 0){
   //     tts.speak("The " + objectName + " is slightly to the left of you.", TextToSpeech.QUEUE_ADD, null);
   //   } else if (location.right <= screenSize.width * 2 / 3 && location.right >= screenSize.width / 3) {
   //     tts.speak("The " + objectName + " is in front of you.", TextToSpeech.QUEUE_ADD, null);
   //   } else {
   //     tts.speak("The " + objectName + " is slightly to the right of you.", TextToSpeech.QUEUE_ADD, null);
   //   }
   // } else {
   //   int pixFromLeft = (int) location.left;
   //   int pixFromRight = screenSize - (int) location.right;
   //   if (pixFromLeft >= pixFromRight*2) {
   //     tts.speak("The " + objectName + " is slightly to the right of you.", TextToSpeech.QUEUE_ADD, null);
   //   } else if (pixFromRight >= pixFromLeft*2) {
   //     tts.speak("The " + objectName + " is slightly to the left of you.", TextToSpeech.QUEUE_ADD, null);
   //   } else {
   //     tts.speak("The " + objectName + " is in front of you.", TextToSpeech.QUEUE_ADD, null);
   //   }
   // }
  }
}

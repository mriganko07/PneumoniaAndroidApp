package com.example.pneumoniaaiapp;


import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import android.graphics.pdf.PdfDocument;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.widget.Toast;

import java.io.FileOutputStream;

import android.widget.ProgressBar;

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_PICK_CODE = 1000;
    private static final int CAMERA_CAPTURE_CODE = 1001;

    private Interpreter pneumoniaTflite;
    private Interpreter preClassifierTflite;

    private ImageView imageView;
    private TextView tvPrediction, tvConfidence;

    private String lastPrediction = "";
    private float lastConfidence = 0f;

    private Bitmap selectedBitmap;
    private String currentPhotoPath;

    private ProgressBar progressConfidence;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // ✅ Step 3: Request runtime permissions
        if (checkSelfPermission(android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 200);
        }
        if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 201);
        }

        imageView = findViewById(R.id.imageView);
        tvPrediction = findViewById(R.id.tvPrediction);
        tvConfidence = findViewById(R.id.tvConfidence);
        progressConfidence = findViewById(R.id.progressConfidence);
        Button btnSelect = findViewById(R.id.btnSelect);
        Button btnCamera = findViewById(R.id.btnCamera);

        // Load both TFLite models
        try {
            pneumoniaTflite = new Interpreter(FileUtil.loadMappedFile(this, "pneumonia_model.tflite"));
            preClassifierTflite = new Interpreter(FileUtil.loadMappedFile(this, "xray_preclassifier.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Gallery selection
        btnSelect.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, IMAGE_PICK_CODE);
        });

        // Camera capture
        btnCamera.setOnClickListener(v -> {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (intent.resolveActivity(getPackageManager()) != null) {
                File photoFile = createImageFile();
                if (photoFile != null) {
                    Uri photoURI = FileProvider.getUriForFile(this,
                            "com.example.pneumoniaaiapp.fileprovider", photoFile);
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                    startActivityForResult(intent, CAMERA_CAPTURE_CODE);
                }
            }
        });

        Button btnExportReport = findViewById(R.id.btnExportReport);
        btnExportReport.setOnClickListener(v -> {
            if (selectedBitmap != null && !lastPrediction.isEmpty()) {
                generatePDFReport(selectedBitmap, lastPrediction, lastConfidence);
            } else {
                Toast.makeText(this, "No prediction available", Toast.LENGTH_SHORT).show();
            }
        });

    }

    private File createImageFile() {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(null);
        File image = null;
        try {
            image = File.createTempFile(imageFileName, ".jpg", storageDir);
            currentPhotoPath = image.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == IMAGE_PICK_CODE && data != null) {
                Uri imageUri = data.getData();
                try {
                    selectedBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                    imageView.setImageBitmap(selectedBitmap);

                    runModelAndDisplay(selectedBitmap); // <-- new helper


                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else if (requestCode == CAMERA_CAPTURE_CODE) {
                File file = new File(currentPhotoPath);
                if (file.exists()) {
                    selectedBitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                    imageView.setImageBitmap(selectedBitmap);

                    runModelAndDisplay(selectedBitmap); // <-- new helper
                } else {
                    tvPrediction.setText("⚠️ Image file not found");
                    tvConfidence.setText("");
                }
            }
        }
    }

    private void generatePDFReport(Bitmap xrayImage, String prediction, float confidence) {
        PdfDocument pdfDocument = new PdfDocument();
        Paint paint = new Paint();

        // Create page info
        PdfDocument.PageInfo pageInfo = new PdfDocument.PageInfo.Builder(595, 842, 1).create();
        PdfDocument.Page page = pdfDocument.startPage(pageInfo);
        Canvas canvas = page.getCanvas();

        // Title
        paint.setTextSize(24);
        paint.setFakeBoldText(true);
        canvas.drawText("Pneumonia Prediction Report", 50, 50, paint);

        // Date & Time
        paint.setTextSize(14);
        paint.setFakeBoldText(false);
        String dateTime = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new java.util.Date());
        canvas.drawText("Generated on: " + dateTime, 50, 90, paint);

        // Prediction + Confidence
        canvas.drawText("Prediction: " + prediction, 50, 140, paint);
        canvas.drawText("Confidence: " + String.format("%.2f%%", confidence * 100), 50, 170, paint);

        // X-ray Image
        if (xrayImage != null) {
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(xrayImage, 400, 400, true);
            canvas.drawBitmap(scaledBitmap, 50, 200, paint);
        }

        pdfDocument.finishPage(page);

        // Save file
        String fileName = "Pneumonia_Report_" + System.currentTimeMillis() + ".pdf";
        File file = new File(getExternalFilesDir(null), fileName);

        try {
            pdfDocument.writeTo(new FileOutputStream(file));
            Toast.makeText(this, "PDF saved: " + file.getAbsolutePath(), Toast.LENGTH_LONG).show();

            // Optionally share
            sharePDF(file);

        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error saving PDF", Toast.LENGTH_SHORT).show();
        }

        pdfDocument.close();
    }


    private void sharePDF(File file) {
        Uri uri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", file);
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("application/pdf");
        shareIntent.putExtra(Intent.EXTRA_STREAM, uri);
        startActivity(Intent.createChooser(shareIntent, "Share Report via"));
    }


    // 🔹 Run X-ray Preclassifier
    private boolean isXray(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        resized.getPixels(intValues, 0, 224, 0, 0, 224, 224);

        for (int pixel : intValues) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            inputBuffer.putFloat(r / 255.0f);
            inputBuffer.putFloat(g / 255.0f);
            inputBuffer.putFloat(b / 255.0f);
        }

        float[][] output = new float[1][1];
        preClassifierTflite.run(inputBuffer, output);

        return output[0][0] > 0.5f; // true if it's an X-ray
    }

    private void runModelAndDisplay(Bitmap bitmap) {
        if (!isXray(bitmap)) {
            tvPrediction.setText("Not an X-ray ❌");
            tvConfidence.setText("");
            return;
        }

        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        resized.getPixels(intValues, 0, 224, 0, 0, 224, 224);

        for (int pixel : intValues) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            inputBuffer.putFloat(r / 255.0f);
            inputBuffer.putFloat(g / 255.0f);
            inputBuffer.putFloat(b / 255.0f);
        }

        float[][] output = new float[1][1];
        pneumoniaTflite.run(inputBuffer, output);

        float confidence = output[0][0];

        if (confidence > 0.5f) {
            lastPrediction = "Pneumonia";
            lastConfidence = confidence;
            tvPrediction.setText("Prediction: Pneumonia");
            tvConfidence.setText("Confidence: " + String.format("%.2f", confidence * 100) + "%");
            progressConfidence.setProgress((int)(confidence * 100)); // <-- add this
        } else {
            lastPrediction = "Normal";
            lastConfidence = 1 - confidence;
            tvPrediction.setText("Prediction: Normal");
            tvConfidence.setText("Confidence: " + String.format("%.2f", (1 - confidence) * 100) + "%");
            progressConfidence.setProgress((int)((1 - confidence) * 100)); // <-- add this
        }
    }
}

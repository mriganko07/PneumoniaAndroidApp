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

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_PICK_CODE = 1000;
    private static final int CAMERA_CAPTURE_CODE = 1001;

    private Interpreter pneumoniaTflite;
    private Interpreter preClassifierTflite;

    private ImageView imageView;
    private TextView tvResult;
    private Bitmap selectedBitmap;
    private String currentPhotoPath;

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
        tvResult = findViewById(R.id.tvResult);
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

                    String prediction = runModel(selectedBitmap);
                    tvResult.setText("Prediction: " + prediction);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else if (requestCode == CAMERA_CAPTURE_CODE) {
                // ✅ Step 4: Debug check if file exists
                File file = new File(currentPhotoPath);
                if (file.exists()) {
                    selectedBitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                    imageView.setImageBitmap(selectedBitmap);

                    String prediction = runModel(selectedBitmap);
                    tvResult.setText("Prediction: " + prediction);
                } else {
                    tvResult.setText("⚠️ Image file not found: " + currentPhotoPath);
                }
            }
        }
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

    // 🔹 Run Pneumonia Model
    private String runModel(Bitmap bitmap) {
        if (!isXray(bitmap)) {
            return "Not an X-ray ❌";
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

        if (output[0][0] > 0.5f) {
            return "X-ray ✅ → Pneumonia";
        } else {
            return "X-ray ✅ → Normal";
        }
    }
}

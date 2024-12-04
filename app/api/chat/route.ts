import { NextResponse } from 'next/server';
import { spawn } from 'child_process';

export async function POST(req: Request) {
  try {
    const { query } = await req.json();

    // Create a promise to handle the Python script execution
    const pythonResponse = await new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [
        'python_Script/main_i_v.py',
        query
      ]);

      let result = '';

      pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python process exited with code ${code}`));
        } else {
          resolve(result);
        }
      });
    });

    return NextResponse.json({ response: pythonResponse });
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
} 
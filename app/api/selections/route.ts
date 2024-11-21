import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function POST(request: Request) {
    const data = await request.json();
    const filePath = path.join(process.cwd(), 'app/issue/selections.json');
  
    let existingData = [];
    if (fs.existsSync(filePath)) {
      const fileContent = fs.readFileSync(filePath, 'utf-8');
      existingData = fileContent ? JSON.parse(fileContent) : [];
    }
    
    existingData.push(data);
    fs.writeFileSync(filePath, JSON.stringify(existingData, null, 2));

    return NextResponse.json({ success: true });
}
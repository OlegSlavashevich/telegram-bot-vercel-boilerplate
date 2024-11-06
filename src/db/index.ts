
import { Db, MongoClient } from 'mongodb';
import * as dotenv from 'dotenv';

dotenv.config();

const mongoClient = new MongoClient(process.env.MONGODB_URI!);
let db: Db;

async function connectToMongo() {
  try {
    await mongoClient.connect();
    db = mongoClient.db('telegram-bot');
    console.log('Connected to MongoDB');
  } catch (error) {
    console.error('Failed to connect to MongoDB:', error);
  }
}

export { db, connectToMongo };

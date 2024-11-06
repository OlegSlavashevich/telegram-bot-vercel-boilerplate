import { Context, Markup, Telegraf } from 'telegraf';

import { VercelRequest, VercelResponse } from '@vercel/node';
import { development, production } from './core';
import { ObjectId } from 'mongodb';
import OpenAI from 'openai';
import { db, connectToMongo } from './db';
import { ChatCompletionContentPart, ChatCompletionMessageParam } from 'openai/resources';
import { randomUUID } from 'crypto';
import { message } from 'telegraf/filters';
import axios from 'axios';

interface UserProfile {
  _id?: ObjectId;
  userId: number;
  username?: string;
  subscription: 'free' | 'premium';
  dailyRequests: number;
  lastResetDate: Date;
  subscriptionExpiryDate?: Date | undefined;
}

interface Invoice {
  _id?: ObjectId;
  invoice_id: string;
  user_id: number;
  amount: number;
  title: string;
  description: string;
  payload: string;
  created_at: Date;
}

interface Payment {
  _id?: ObjectId;
  payment_id: string;
  invoice_id: string;
  user_id: number;
  amount: number;
  status: 'pending' | 'completed' | 'refunded';
  created_at: Date;
}


const BOT_TOKEN = process.env.BOT_TOKEN || '';
const ENVIRONMENT = process.env.NODE_ENV || '';

const bot = new Telegraf(BOT_TOKEN);

const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
  baseURL: 'https://openrouter.ai/api/v1',
  defaultHeaders: {
    'X-Title': 'Telegram GPT Bot'
  }
});

const MAX_CONTEXT_MESSAGES = 8;
const STREAM_CHUNK_SIZE = 100;
const FREE_DAILY_LIMIT = 10;
const PREMIUM_DAILY_LIMIT = 100;
const PREMIUM_PRICE = 1; // Цена премиум подписки в звездах

// Добавляем константу для максимального размера файла (10 МБ в байтах)
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

async function getChatHistory(userId: number): Promise<Array<{ role: 'system' | 'user' | 'assistant', content: string }>> {
  const chats = db.collection('chats');
  const history = await chats.find({ userId }).sort({ timestamp: -1 }).limit(MAX_CONTEXT_MESSAGES).toArray();
  return history.map(msg => ({ 
    role: msg.role as 'system' | 'user' | 'assistant', 
    content: msg.content 
  })).reverse();
}

async function saveChatMessage(userId: number, role: 'system' | 'user' | 'assistant', content: string) {
  const chats = db.collection('chats');
  await chats.insertOne({ 
    userId, 
    role, 
    content, 
    timestamp: new Date() 
  });
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

async function sendToOpenRouterStream(userId: number, prompt: string | Array<ChatCompletionContentPart>, ctx: Context): Promise<void> {
  try {
    // Получаем профиль пользователя для проверки подписки
    const user = await getUserProfile(userId);
    const model = user.subscription === 'premium' ? 'gpt-4o' : 'gpt-4o-mini';

    const history = await getChatHistory(userId);
    const messages: ChatCompletionMessageParam[] = [
      ...history.map(msg => ({
        role: msg.role as 'system' | 'user' | 'assistant',
        content: msg.content
      })),
      { role: 'user', content: prompt }
    ];

    let totalResponseTokens = 0;
    let responseChunk = '';
    let sentMessage: any;

    const stream = await openai.chat.completions.create({
      model: model, // Используем модель в зависимости от подписки
      messages: messages,
      stream: true,
      temperature: 0.7,
      max_tokens: user.subscription === 'premium' ? 4000 : 2000, // Разные лимиты токенов
    });

    for await (const part of stream) {
      const content = part.choices[0]?.delta?.content || '';
      responseChunk += content;

      if (responseChunk.length >= STREAM_CHUNK_SIZE || content.includes('\n')) {
        if (sentMessage) {
          await ctx.telegram.editMessageText(
            ctx.chat!.id,
            sentMessage.message_id,
            undefined,
            sentMessage.text + responseChunk
          );
          sentMessage.text += responseChunk;
        } else {
          sentMessage = await ctx.reply(responseChunk);
        }
        responseChunk = '';
      }

      if (part.usage?.total_tokens) {
        totalResponseTokens = part.usage.total_tokens;
      }
    }

    if (responseChunk) {
      if (sentMessage) {
        await ctx.telegram.editMessageText(
          ctx.chat!.id,
          sentMessage.message_id,
          undefined,
          sentMessage.text + responseChunk
        );
      } else {
        await ctx.reply(responseChunk);
      }
    }

    const fullResponse = sentMessage ? sentMessage.text + responseChunk : responseChunk;
    await saveChatMessage(userId, 'assistant', fullResponse);

    if (totalResponseTokens === 0) {
      totalResponseTokens = estimateTokens(fullResponse);
    }

    await updateTokenUsage(userId, estimateTokens(prompt as string), totalResponseTokens);
  } catch (error) {
    console.error('Error calling OpenRouter:', error);
    await ctx.reply('Sorry, there was an error processing your request.');
  }
}

async function updateTokenUsage(userId: number, inputTokens: number, outputTokens: number) {
  const stats = db.collection('user_stats');
  await stats.updateOne(
    { userId },
    { 
      $inc: { 
        totalInputTokens: inputTokens, 
        totalOutputTokens: outputTokens 
      } 
    },
    { upsert: true }
  );
}

async function checkAndUpdateSubscriptionStatus(userId: number, ctx?: Context): Promise<void> {
  const users = db.collection<UserProfile>('users');
  const user = await users.findOne({ userId });
  
  if (!user) return;

  if (user.subscription === 'premium' && user.subscriptionExpiryDate) {
    const now = new Date();
    if (now > user.subscriptionExpiryDate) {
      // Подписка истекла, сбрасываем на free
      await users.updateOne(
        { userId },
        { 
          $set: { 
            subscription: 'free',
            subscriptionExpiryDate: undefined
          } 
        }
      );

      // Если передан контекст, отправляем уведомление
      if (ctx) {
        await ctx.reply(
          'Ваша премиум подписка истекла! 😢\n' +
          'Чтобы продолжить пользоваться расширенными возможностями, ' +
          'пожалуйста, обновите подписку командой /pay',
          Markup.inlineKeyboard([
            Markup.button.callback('Обновить подписку', 'buy_premium')
          ])
        );
      }
    }
  }
}

async function getUserProfile(userId: number, username?: string): Promise<UserProfile> {
  await checkAndUpdateSubscriptionStatus(userId);
  
  const users = db.collection<UserProfile>('users');
  let user = await users.findOne({ userId });
  
  if (!user) {
    const newUser: UserProfile = {
      userId,
      username,
      subscription: 'free',
      dailyRequests: 0,
      lastResetDate: new Date(),
    };
    const result = await users.insertOne(newUser);
    newUser._id = result.insertedId;
    return newUser;
  }
  
  if (username && username !== user.username) {
    await users.updateOne(
      { userId },
      { $set: { username } }
    );
    user.username = username;
  }
  
  return user;
}

async function updateUserRequests(userId: number): Promise<boolean> {
  await checkAndUpdateSubscriptionStatus(userId);
  
  const users = db.collection<UserProfile>('users');
  const user = await getUserProfile(userId);
  
  const now = new Date();
  const resetTime = new Date(user.lastResetDate.getTime() + 24 * 60 * 60 * 1000); // lastResetDate + 24 часа
  
  if (now >= resetTime) {
    // Прошло 24 час с момента последнего сброса
    await users.updateOne(
      { userId }, 
      { 
        $set: { 
          dailyRequests: 1, 
          lastResetDate: now 
        } 
      }
    );
    return true;
  }
  
  const limit = user.subscription === 'premium' ? PREMIUM_DAILY_LIMIT : FREE_DAILY_LIMIT;
  if (user.dailyRequests >= limit) {
    return false;
  }
  
  await users.updateOne({ userId }, { $inc: { dailyRequests: 1 } });
  return true;
}

async function resetUserContext(userId: number) {
  const chats = db.collection('chats');
  await chats.deleteMany({ userId });
}

async function cancelSubscription(userId: number): Promise<boolean> {
  const users = db.collection<UserProfile>('users');
  const result = await users.updateOne(
    { userId },
    { 
      $set: { 
        subscription: 'free',
        subscriptionExpiryDate: undefined
      } 
    }
  );
  return result.modifiedCount > 0;
}

async function createInvoice(userId: number): Promise<Invoice> {
  const invoices = db.collection<Invoice>('invoices');
  const invoice: Invoice = {
    invoice_id: randomUUID(),
    user_id: userId,
    amount: PREMIUM_PRICE,
    title: 'Премиум подписка',
    description: 'Премиум подписка на 30 дней',
    payload: randomUUID(),
    created_at: new Date()
  };
  
  await invoices.insertOne(invoice);
  return invoice;
}

async function createPayment(paymentData: any): Promise<Payment> {
  const payments = db.collection<Payment>('payments');
  const payment: Payment = {
    payment_id: paymentData.telegram_payment_charge_id,
    invoice_id: paymentData.invoice_id,
    user_id: paymentData.user_id,
    amount: paymentData.total_amount,
    status: 'completed',
    created_at: new Date()
  };
  
  await payments.insertOne(payment);
  return payment;
}

// Добавляем новую функцию для проверки премиум статуса
async function checkPremiumStatus(userId: number, ctx: Context): Promise<boolean> {
  const user = await getUserProfile(userId);
  
  if (user.subscription === 'premium') {
    const expiryDate = user.subscriptionExpiryDate;
    await ctx.reply(
      'У вас уже есть активная премиум подписка!' +
      (expiryDate ? `\nСрок действия: до ${expiryDate.toLocaleString('ru-RU', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        timeZone: 'Europe/Moscow'
      })}` : '')
    );
    return true;
  }
  return false;
}

// Добавляем функцию для отправки инвойса
async function sendSubscriptionInvoice(userId: number, ctx: Context): Promise<void> {
  const invoice = await createInvoice(userId);
  
  try {
    await ctx.telegram.sendInvoice(ctx.chat!.id, {
      title: invoice.title,
      description: invoice.description,
      payload: invoice.payload,
      provider_token: "",
      currency: 'XTR',
      prices: [{ 
        label: 'Премиум подписка', 
        amount: PREMIUM_PRICE
      }],
    });
  } catch (error) {
    console.error('Error sending invoice:', error);
    await ctx.reply('Произошла ошибка при создании счета. Пожалуйста, попробуйте позже.');
  }
}

bot.command('start', async (ctx) => {
  const username = ctx.from.username;
  await getUserProfile(ctx.from.id, username);
  const welcomeMessage = `
Добро пожаловать в AI бота!

Наши тарифы:
1. Бесплатный: ${FREE_DAILY_LIMIT} текстовых запросов в день
2. Премиум: ${PREMIUM_DAILY_LIMIT} запросов в день

Премиум возможности:
• Анализ изображений и фотографий
• Анализ документов (PDF, DOCX, TXT, CSV, JSON)
• Больше токенов на ответ
• Доступ к более мощной модели

Цена премиум подписки: ${PREMIUM_PRICE} ⭐

Используйте команду /pay для покупки премиум подписки.
  `;

  const keyboard = Markup.keyboard([
    [Markup.button.text('/profile'), Markup.button.text('/pay')],
    [Markup.button.text('/reset'), Markup.button.text('/help')]
  ]).resize();

  ctx.reply(welcomeMessage, keyboard);
});

bot.command('profile', async (ctx) => {
  const userId = ctx.from.id;
  const username = ctx.from.username;
  await checkAndUpdateSubscriptionStatus(userId, ctx);
  const user = await getUserProfile(userId, username);
  const nextReset = new Date(user.lastResetDate);
  nextReset.setDate(nextReset.getDate() + 1);
  
  let subscriptionInfo = `Подписка: ${user.subscription}`;
  if (user.subscription === 'premium') {
    subscriptionInfo += '\nДля отмены подписки используйте команду /cancel_subscription';
  } else {
    subscriptionInfo += '\nДля покупки премиум подписки нажмите /pay';
  }
  
  const profileMessage = `
Это ваш профиль (/profile).
ID: ${userId}
Username: ${user.username || 'не указан'}
${subscriptionInfo}

Лимиты
осталось ${user.subscription === 'premium' ? PREMIUM_DAILY_LIMIT - user.dailyRequests : FREE_DAILY_LIMIT - user.dailyRequests}/${user.subscription === 'premium' ? PREMIUM_DAILY_LIMIT : FREE_DAILY_LIMIT} сегодня
Обновление лимитов: ${nextReset.toLocaleString('ru-RU', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Moscow' })} (мск)
  `;
  
  ctx.reply(profileMessage);
});

bot.command('pay', async (ctx) => {
  const userId = ctx.from!.id;
  const hasPremium = await checkPremiumStatus(userId, ctx);
  if (!hasPremium) {
    await sendSubscriptionInvoice(userId, ctx);
  }
});

bot.command('reset', async (ctx) => {
  const userId = ctx.from.id;
  await resetUserContext(userId);
  ctx.reply('Контекст вашего разговора был сброшен.');
});

bot.command('help', (ctx) => {
  const helpMessage = `
Доступные команды:
/start - Перезапустить бота и посмотреть тарифы
/profile - Посмотреть ваш профиль и статистику
/pay - Купить премиум подписку
/cancel_subscription - Отменить премиум подписку
/reset - Сбросить контекст разговора
/help - Показать это сообщение помощи
  `;

  const keyboard = Markup.keyboard([
    [Markup.button.text('/profile'), Markup.button.text('/pay')],
    [Markup.button.text('/cancel_subscription'), Markup.button.text('/reset')],
    [Markup.button.text('/help')]
  ]).resize();

  ctx.reply(helpMessage, keyboard);
});

bot.command('cancel_subscription', async (ctx) => {
  const userId = ctx.from.id;
  const user = await getUserProfile(userId);

  if (user.subscription !== 'premium') {
    await ctx.reply('У вас нет активной премиум подписки.');
    return;
  }

  const cancelled = await cancelSubscription(userId);
  if (cancelled) {
    await ctx.reply('Ваша премиум подписка отменена. Изменения вступят в силу при следующем обновлении лимитов.');
  } else {
    await ctx.reply('Произошла ошибка при отмене подписки. Пожалуйста, попробуйте позже или обратитесь в поддержку.');
  }
});

bot.on(message('text'), async (ctx) => {
  const userId = ctx.from.id;
  const username = ctx.from.username;
  const userMessage = ctx.message.text;
  console.log('Received message:', userMessage);

  try {
    await checkAndUpdateSubscriptionStatus(userId, ctx);
    const canMakeRequest = await updateUserRequests(userId);
    if (!canMakeRequest) {
      const user = await getUserProfile(userId, username);
      if (user.subscription === 'free') {
        await ctx.reply('Вы достигли дневного лимита бесплатных запросов. Хотите купить премиум подписку?', 
          Markup.inlineKeyboard([Markup.button.callback('Купить премиум', 'buy_premium')]));
      } else {
        await ctx.reply('Вы достигли дневного лимита запросов. Попробуйте снова завтра.');
      }
      return;
    }

    await saveChatMessage(userId, 'user', userMessage);
    await sendToOpenRouterStream(userId, userMessage, ctx);
  } catch (error) {
    console.error('Error processing message:', error);
    await ctx.reply('Извините, произошла ошибка при обработке вашего сообщения.');
  }
});

bot.on('pre_checkout_query', async (ctx) => {
  const { id, invoice_payload } = ctx.preCheckoutQuery;
  try {
    const invoice = await db.collection<Invoice>('invoices').findOne({ payload: invoice_payload });
    if (!invoice) {
      await ctx.answerPreCheckoutQuery(false, 'Инвойс не найден');
      return;
    }
    await ctx.answerPreCheckoutQuery(true);
  } catch (error) {
    console.error('Pre-checkout error:', error);
    await ctx.answerPreCheckoutQuery(false, 'Произошла ошибка при проверке платежа');
  }
});

bot.on('successful_payment', async (ctx) => {
  const payment = ctx.message?.successful_payment;
  if (!payment) return;

  try {
    const invoice = await db.collection<Invoice>('invoices').findOne({ 
      payload: payment.invoice_payload 
    });
    
    if (!invoice) {
      await ctx.reply('Ошибка при обработке платежа. Пожалуйста, обратитесь в поддержку.');
      return;
    }

    await createPayment({
      payment_id: payment.telegram_payment_charge_id,
      invoice_id: invoice.invoice_id,
      user_id: ctx.from.id,
      amount: payment.total_amount,
      status: 'completed',
      created_at: new Date()
    });
    
    // Активируем премиум подписку
    const users = db.collection<UserProfile>('users');
    const expiryDate = new Date();
    expiryDate.setDate(expiryDate.getDate() + 30);

    await users.updateOne(
      { userId: ctx.from.id },
      { 
        $set: { 
          subscription: 'premium',
          subscriptionExpiryDate: expiryDate
        } 
      }
    );

    await ctx.reply('Спасибо за покупку! Ваша ремиум подписка активирована на 30 дней.');
  } catch (error) {
    console.error('Payment processing error:', error);
    await ctx.reply('Произошла ошибка при обработке платежа. Наша команда уже работает над этим.');
  }
});

bot.action('buy_premium', async (ctx) => {
  await ctx.answerCbQuery();
  const userId = ctx.from!.id;
  const hasPremium = await checkPremiumStatus(userId, ctx);
  if (!hasPremium) {
    await sendSubscriptionInvoice(userId, ctx);
  }
});

bot.telegram.setMyCommands([
  { command: 'start', description: 'Перезапустить бота и посмотреть тарифы' },
  { command: 'profile', description: 'Посмотреть ваш профиль и статистику' },
  { command: 'pay', description: 'Купить премиум подписку' },
  { command: 'cancel_subscription', description: 'Отменить премиум подписку' },
  { command: 'reset', description: 'Сбросить контекст разговора' },
  { command: 'help', description: 'Показать сообщение помощи' }
]);

//prod mode (Vercel)
export const startVercel = async (req: VercelRequest, res: VercelResponse) => {
  await connectToMongo();
  await production(req, res, bot);
};
//dev mode
ENVIRONMENT !== 'production' && development(bot);

// Добавляем обработчик для документов и фото
bot.on(message('document'), async (ctx) => {
  console.log('document upload');
  await ctx.reply('Извините, фича пока не доступна.');
  return;

  // const userId = ctx.from.id;
  // const username = ctx.from.username;
  // const user = await getUserProfile(userId, username);

  // // Ранняя проверка на премиум подписку
  // if (user.subscription !== 'premium') {
  //   await ctx.reply(
  //     'Анализ файлов доступен только для премиум пользователей.\n' +
  //     'Хотите получить доступ к этой функции?',
  //     Markup.inlineKeyboard([
  //       Markup.button.callback('Купить премиум', 'buy_premium')
  //     ])
  //   );
  //   return;
  // }

  // const document = ctx.message.document;
  
  // // Проверка размера файла
  // if (document.file_size && document.file_size > MAX_FILE_SIZE) {
  //   await ctx.reply('Извините, но размер файла превышает максимально допустимый (10 МБ).');
  //   return;
  // }

  // const caption = ctx.message.caption || '';

  // try {
  //   await checkAndUpdateSubscriptionStatus(userId, ctx);
  //   const canMakeRequest = await updateUserRequests(userId);
  //   if (!canMakeRequest) {
  //     await ctx.reply('Вы достигли дневного лимита запросов. Попробуйте снова завтра.');
  //     return;
  //   }

  //   // Получаем информацию о файле
  //   const file = await ctx.telegram.getFile(document.file_id);
  //   const fileUrl = `https://api.telegram.org/file/bot${BOT_TOKEN}/${file.file_path}`;

  //   // Сохраняем в базу информацию о файле и caption
  //   const fileInfo = `[Файл: ${document.file_name} (${document.mime_type})]`;
  //   const messageForHistory = caption ? `${caption}` : fileInfo;
  //   await saveChatMessage(userId, 'user', messageForHistory);

  //   // Загружаем и обрабатываем содержимое файла
  //   const fileContent = await processFileContent(fileUrl, document.file_name as string);

  //   // Формируем промпт с информацией о файле и его содержимым
  //   const prompt = caption 
  //     ? `User Prompt: ${caption}\n\nFile Content: ${fileContent}`
  //     : `Please analyze this file: ${fileContent}`;

  //   // Отправляем в модель
  //   const response = await sendToOpenRouterStream(userId, prompt, ctx);
  // } catch (error) {
  //   console.error('Error processing file:', error);
  //   await ctx.reply('Извините, произошла ошибка при обработке вашего файла.');
  // }
});

bot.on(message('photo'), async (ctx) => {
  const userId = ctx.from.id;
  const username = ctx.from.username;
  const user = await getUserProfile(userId, username);

  // Ранняя проверка на премиум подписку
  if (user.subscription !== 'premium') {
    await ctx.reply(
      'Анализ изображений доступен только для премиум пользователей.\n' +
      'Хотите получить доступ к этой функции?',
      Markup.inlineKeyboard([
        Markup.button.callback('Купить премиум', 'buy_premium')
      ])
    );
    return;
  }

  const photos = ctx.message.photo;
  const largestPhoto = photos[photos.length - 1];
  
  // Проверка размера фото
  if (largestPhoto.file_size && largestPhoto.file_size > MAX_FILE_SIZE) {
    await ctx.reply('Извините, но размер изображения превышает максимально допустимый (10 МБ).');
    return;
  }

  const caption = ctx.message.caption || '';

  try {
    await checkAndUpdateSubscriptionStatus(userId, ctx);
    const canMakeRequest = await updateUserRequests(userId);
    if (!canMakeRequest) {
      await ctx.reply('Вы достигли дневного лимита запросов. Попробуйте снова завтра.');
      return;
    }

    // Берем фото максимального размера
    const photo = photos[photos.length - 1];
    const file = await ctx.telegram.getFile(photo.file_id);
    const fileUrl = `https://api.telegram.org/file/bot${BOT_TOKEN}/${file.file_path}`;

    // Сохраняем в базу информацию о фото и caption
    const messageForHistory = caption ? caption: "было отправлено фото";
    await saveChatMessage(userId, 'user', messageForHistory);

    console.log(fileUrl);

    // Формируем промпт
    const prompt = [
      { type: "text", text: caption },
      {
        type: "image_url",
        image_url: {
          "url": fileUrl,
        },
      },
    ] as ChatCompletionContentPart[];

    // Отправляем в модель
    await sendToOpenRouterStream(userId, prompt, ctx);
  } catch (error) {
    console.error('Error processing photo:', error);
    await ctx.reply('Извините, произошла ошибка при обработке вашей фотографии.');
  }
});

// Добавляем функцию для загрузки и обработки файла
async function processFileContent(fileUrl: string, fileName: string): Promise<string> {
  try {
    // Загружаем файл в память
    const response = await axios.get(fileUrl, { responseType: 'arraybuffer' });
    const buffer = Buffer.from(response.data);
    
    // Определяем тип файла по расширению
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    // Обрабатываем содержимое в зависимости от типа файла
    switch (extension) {
      case 'txt':
        return buffer.toString('utf-8');
        
      case 'json':
        return JSON.stringify(JSON.parse(buffer.toString('utf-8')), null, 2);
        
      case 'csv':
        return buffer.toString('utf-8');
        
      case 'pdf':
        // Для PDF можно использовать pdf-parse
        const pdfParse = require('pdf-parse');
        const data = await pdfParse(buffer);
        return data.text;
        
      case 'docx':
        // Для DOCX можно использовать mammoth
        const mammoth = require('mammoth');
        const result = await mammoth.extractRawText({ buffer });
        return result.value;
        
      default:
        return buffer.toString('utf-8');
    }
  } catch (error) {
    console.error('Error processing file:', error);
    throw new Error('Failed to process file content');
  }
}


import { Context, Markup, Telegraf } from 'telegraf';

import { VercelRequest, VercelResponse } from '@vercel/node';
import { development, production } from './core';
import { ObjectId } from 'mongodb';
import OpenAI from 'openai';
import { db, connectToMongo } from './db';
import { ChatCompletionMessageParam } from 'openai/resources';
import { randomUUID } from 'crypto';
import { message } from 'telegraf/filters';

interface UserProfile {
  _id?: ObjectId;
  userId: number;
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
const PREMIUM_PRICE = 500; // Цена премиум подписки в рублях

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

async function sendToOpenRouterStream(userId: number, prompt: string, ctx: Context): Promise<void> {
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

    await updateTokenUsage(userId, estimateTokens(prompt), totalResponseTokens);
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

async function getUserProfile(userId: number): Promise<UserProfile> {
  await checkAndUpdateSubscriptionStatus(userId);
  
  const users = db.collection<UserProfile>('users');
  let user = await users.findOne({ userId });
  if (!user) {
    const newUser: UserProfile = {
      userId,
      subscription: 'free',
      dailyRequests: 0,
      lastResetDate: new Date(),
    };
    const result = await users.insertOne(newUser);
    newUser._id = result.insertedId;
    return newUser;
  }
  return user;
}

async function updateUserRequests(userId: number): Promise<boolean> {
  await checkAndUpdateSubscriptionStatus(userId);
  
  const users = db.collection<UserProfile>('users');
  const user = await getUserProfile(userId);
  
  const today = new Date();
  if (user.lastResetDate.toDateString() !== today.toDateString()) {
    await users.updateOne({ userId }, { $set: { dailyRequests: 1, lastResetDate: today } });
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

bot.command('start', (ctx) => {
  const welcomeMessage = `
Добро пожаловать в AI бота!

Наши тарифы:
1. Бесплатный: ${FREE_DAILY_LIMIT} генераций в день
2. Премиум: ${PREMIUM_DAILY_LIMIT} генераций в день

Цена премиум подписки: ${PREMIUM_PRICE} руб/месяц

Используйте команду /pay для покупки премиум пописки.
  `;

  const keyboard = Markup.keyboard([
    [Markup.button.text('/profile'), Markup.button.text('/pay')],
    [Markup.button.text('/reset'), Markup.button.text('/help')]
  ]).resize();

  ctx.reply(welcomeMessage, keyboard);
});

bot.command('profile', async (ctx) => {
  const userId = ctx.from.id;
  await checkAndUpdateSubscriptionStatus(userId, ctx);
  const user = await getUserProfile(userId);
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
${subscriptionInfo}

Лимиты
осталось ${user.subscription === 'premium' ? PREMIUM_DAILY_LIMIT - user.dailyRequests : FREE_DAILY_LIMIT - user.dailyRequests}/${user.subscription === 'premium' ? PREMIUM_DAILY_LIMIT : FREE_DAILY_LIMIT} сегодня
Обновление лимитов: ${nextReset.toLocaleString('ru-RU', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Moscow' })} (мск)
  `;
  
  ctx.reply(profileMessage);
});

bot.command('pay', async (ctx) => {
  const userId = ctx.from!.id;
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
        amount: 1
      }],
    });
  } catch (error) {
    console.error('Error sending invoice:', error);
    await ctx.reply('Произошла ошибка при создании счета. Пожалуйста, попробуйте позже.');
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
  const userMessage = ctx.message.text;
  console.log('Received message:', userMessage);

  try {
    await checkAndUpdateSubscriptionStatus(userId, ctx);
    const canMakeRequest = await updateUserRequests(userId);
    if (!canMakeRequest) {
      const user = await getUserProfile(userId);
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
        amount: 1
      }],
    });
  } catch (error) {
    console.error('Error sending invoice:', error);
    await ctx.reply('Произошла ошибка при создании счета. Пожалуйста, попробуйте позже.');
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

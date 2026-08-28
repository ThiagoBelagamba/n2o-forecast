import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors({
    origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  });
  await app.listen(3001);
  console.log('API n2o-forecast em http://localhost:3001');
}
bootstrap();

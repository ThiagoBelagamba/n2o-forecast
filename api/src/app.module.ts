import { Module } from '@nestjs/common';
import { ServeStaticModule } from '@nestjs/serve-static';
import { ResultsModule } from './results/results.module';
import { graficosRoot } from './paths';

@Module({
  imports: [
    ServeStaticModule.forRoot({
      rootPath: graficosRoot(),
      serveRoot: '/static/graficos',
    }),
    ResultsModule,
  ],
})
export class AppModule {}

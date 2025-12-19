import numpy as np
import matplotlib.pyplot as plt

def simple_perceptron_forward(x, W1, W2):
    """
    Прямой проход через простой перцептрон из презентации
    """
    # Первый слой (линейная комбинация)
    L2 = np.dot(W1, x)
    
    # Функция активации ReLU
    L2_out = np.maximum(0, L2)
    
    # Выходной слой
    output = np.dot(W2.T, L2_out)
    
    return output[0], L2, L2_out  # Возвращаем скаляр вместо массива

def compute_error(output, target):
    """
    Вычисление квадратичной ошибки (MSE)
    """
    return 0.5 * (output - target)**2

def gradient_descent_step(x, target, W1, W2, learning_rate=0.1):
    """
    Один шаг градиентного спуска
    """
    # Прямой проход
    output, L2, L2_out = simple_perceptron_forward(x, W1, W2)
    
    # Вычисление ошибки
    error = compute_error(output, target)
    
    # Вычисление градиентов (обратное распространение)
    dE_dO = output - target  # ∂E/∂O
    
    # Градиенты для выходного слоя
    dE_dW2 = dE_dO * L2_out  # ∂E/∂W2 = ∂E/∂O * ∂O/∂W2
    
    # Градиенты для скрытого слоя
    # δ2 = ∂E/∂L2_out = ∂E/∂O * W2
    delta2 = dE_dO * W2.flatten()
    
    # Учитываем производную ReLU: ∂L2_out/∂L2 = 1 если L2 > 0, иначе 0
    relu_derivative = (L2 > 0).astype(float)
    delta2_in = delta2 * relu_derivative
    
    # Градиенты для весов W1
    dE_dW1 = np.outer(delta2_in, x)  # ∂E/∂W1 = δ2_in * x^T
    
    # Обновление весов
    W2_new = W2 - learning_rate * dE_dW2.reshape(-1, 1)
    W1_new = W1 - learning_rate * dE_dW1
    
    return W1_new, W2_new, error, output

def train_perceptron(epochs=20, learning_rate=0.1):
    """
    Обучение перцептрона и запись истории ошибок
    """
    # Инициализация как в презентации
    x = np.array([0.5, 0.7])  # Входные данные
    target = 1.0              # Ожидаемый выход
    
    # Начальные веса (как в презентации)
    W1 = np.array([[1, 3], [-2, 4]])  # Веса первого слоя 2x2
    W2 = np.array([[0.5], [0.3]])     # Веса выходного слоя 2x1
    
    # История для визуализации
    errors = []
    outputs = []
    W1_history = [W1.copy()]
    W2_history = [W2.copy()]
    
    # Начальный прямой проход
    initial_output, _, _ = simple_perceptron_forward(x, W1, W2)
    initial_error = compute_error(initial_output, target)
    errors.append(initial_error)
    outputs.append(initial_output)
    
    print(f"Начальная ошибка: {initial_error:.6f}")
    print(f"Начальный выход: {initial_output:.6f}")
    print(f"Начальные веса W1:\n{W1}")
    print(f"Начальные веса W2:\n{W2}")
    print("-" * 50)
    
    # Обучение
    for epoch in range(epochs):
        # Сохраняем веса до обновления
        W1_old = W1.copy()
        W2_old = W2.copy()
        
        # Выполняем шаг градиентного спуска
        W1, W2, error, output = gradient_descent_step(x, target, W1, W2, learning_rate)
        
        errors.append(error)
        outputs.append(output)
        W1_history.append(W1.copy())
        W2_history.append(W2.copy())
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Эпоха {epoch + 1}:")
            print(f"  Ошибка: {error:.6f}")
            print(f"  Выход: {output:.6f}")
            print(f"  W1[0,0]: {W1[0,0]:.6f} (было: {W1_old[0,0]:.6f})")
            print(f"  W2[0]: {W2[0,0]:.6f} (было: {W2_old[0,0]:.6f})")
            print("-" * 30)
    
    return errors, outputs, W1_history, W2_history

def visualize_training(errors, outputs):
    """
    Визуализация процесса обучения
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Функция ошибки по эпохам
    axes[0, 0].plot(range(len(errors)), errors, 'b-', linewidth=2, marker='o', markersize=5)
    axes[0, 0].set_xlabel('Эпоха', fontsize=12)
    axes[0, 0].set_ylabel('Ошибка (MSE)', fontsize=12)
    axes[0, 0].set_title('График функции ошибки во время обучения', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Добавляем метки для некоторых точек
    for i in [0, 5, 10, 15, 20]:
        if i < len(errors):
            axes[0, 0].annotate(f'{errors[i]:.4f}', 
                               xy=(i, errors[i]), 
                               xytext=(i, errors[i] + 0.02),
                               ha='center', fontsize=9)
    
    # График 2: Выход сети по эпохам
    axes[0, 1].plot(range(len(outputs)), outputs, 'g-', linewidth=2, marker='s', markersize=5)
    axes[0, 1].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Целевое значение = 1.0')
    axes[0, 1].set_xlabel('Эпоха', fontsize=12)
    axes[0, 1].set_ylabel('Выход сети', fontsize=12)
    axes[0, 1].set_title('Изменение выхода сети во время обучения', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # График 3: Логарифмический масштаб ошибки
    axes[1, 0].semilogy(range(len(errors)), errors, 'r-', linewidth=2, marker='^', markersize=5)
    axes[1, 0].set_xlabel('Эпоха', fontsize=12)
    axes[1, 0].set_ylabel('log(Ошибка)', fontsize=12)
    axes[1, 0].set_title('Ошибка в логарифмическом масштабе', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Скорость изменения ошибки
    error_changes = np.diff(errors)
    axes[1, 1].plot(range(1, len(errors)), error_changes, color='purple', linewidth=2, marker='d', markersize=5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Эпоха', fontsize=12)
    axes[1, 1].set_ylabel('Δ Ошибки', fontsize=12)
    axes[1, 1].set_title('Изменение ошибки между эпохами', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Добавляем отрицательные значения в явном виде
    if len(error_changes) > 0:
        axes[1, 1].fill_between(range(1, len(errors)), error_changes, 0, 
                               where=(error_changes < 0), color='green', alpha=0.3, label='Улучшение')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.suptitle('Процесс обучения нейронной сети (пример из презентации)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()

def visualize_3d_error_surface():
    """
    Дополнительная визуализация: 3D поверхность ошибки
    (для одного параметра W2[0])
    """
    # Фиксируем все параметры кроме W2[0]
    x = np.array([0.5, 0.7])
    target = 1.0
    W1 = np.array([[1, 3], [-2, 4]])
    
    # Создаем сетку значений для W2[0]
    w2_values = np.linspace(-1, 2, 50)
    errors = []
    
    for w2 in w2_values:
        W2 = np.array([[w2], [0.3]])  # Меняем только первый вес
        output, _, _ = simple_perceptron_forward(x, W1, W2)
        error = compute_error(output, target)
        errors.append(error)
    
    # График поверхности ошибки
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(w2_values, errors, 'b-', linewidth=3)
    
    # Отмечаем начальное значение и оптимальное
    initial_w2 = 0.5
    initial_output, _, _ = simple_perceptron_forward(x, W1, np.array([[initial_w2], [0.3]]))
    initial_error = compute_error(initial_output, target)
    
    ax.plot(initial_w2, initial_error, 'ro', markersize=10, label=f'Начальная точка (W2[0]={initial_w2})')
    
    # Находим минимум
    min_idx = np.argmin(errors)
    ax.plot(w2_values[min_idx], errors[min_idx], 'g*', markersize=15, 
            label=f'Минимум (W2[0]={w2_values[min_idx]:.3f})')
    
    ax.set_xlabel('Вес W2[0]', fontsize=12)
    ax.set_ylabel('Ошибка', fontsize=12)
    ax.set_title('Зависимость ошибки от веса W2[0] (W1 и W2[1] фиксированы)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Основная функция
    """
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ НЕЙРОННОЙ СЕТИ")
    print("Из примера в презентации 'Нейронные сети'")
    print("=" * 60)
    print("\nПараметры обучения:")
    print("- Входные данные: x = [0.5, 0.7]")
    print("- Целевое значение: 1.0")
    print("- Начальные веса: W1 = [[1, 3], [-2, 4]], W2 = [[0.5], [0.3]]")
    print("- Скорость обучения: 0.1")
    print("- Количество эпох: 20")
    print("=" * 60)
    
    # Обучаем сеть
    errors, outputs, W1_history, W2_history = train_perceptron(epochs=20, learning_rate=0.1)
    
    # Основная визуализация
    visualize_training(errors, outputs)
    
    # Дополнительная 3D визуализация
    visualize_3d_error_surface()
    
    # Вывод итогов
    print("\n" + "=" * 60)
    print("ИТОГИ ОБУЧЕНИЯ:")
    print(f"Начальная ошибка: {errors[0]:.6f}")
    print(f"Финальная ошибка: {errors[-1]:.6f}")
    print(f"Улучшение: {((errors[0] - errors[-1]) / errors[0] * 100):.1f}%")
    print(f"Начальный выход: {outputs[0]:.6f}")
    print(f"Финальный выход: {outputs[-1]:.6f}")
    print(f"Целевое значение: 1.000000")
    print("=" * 60)
    
    # Таблица значений ошибки для первых 5 эпох
    print("\nДетали по эпохам (первые 5):")
    print("Эпоха | Ошибка    | Выход     | Δ ошибки")
    print("-" * 40)
    for i in range(min(6, len(errors))):
        if i == 0:
            delta = 0
        else:
            delta = errors[i] - errors[i-1]
        print(f"{i:5d} | {errors[i]:.6f} | {outputs[i]:.6f} | {delta:+.6f}")

if __name__ == "__main__":
    main()